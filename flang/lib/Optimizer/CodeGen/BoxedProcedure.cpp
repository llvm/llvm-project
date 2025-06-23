//===-- BoxedProcedure.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

namespace fir {
#define GEN_PASS_DEF_BOXEDPROCEDUREPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-procedure-pointer"

using namespace fir;

namespace {
/// Options to the procedure pointer pass.
struct BoxedProcedureOptions {
  // Lower the boxproc abstraction to function pointers and thunks where
  // required.
  bool useThunks = true;
};

/// This type converter rewrites all `!fir.boxproc<Func>` types to `Func` types.
class BoxprocTypeRewriter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::convertType;

  /// Does the type \p ty need to be converted?
  /// Any type that is a `!fir.boxproc` in whole or in part will need to be
  /// converted to a function type to lower the IR to function pointer form in
  /// the default implementation performed in this pass. Other implementations
  /// are possible, so those may convert `!fir.boxproc` to some other type or
  /// not at all depending on the implementation target's characteristics and
  /// preference.
  bool needsConversion(mlir::Type ty) {
    if (mlir::isa<BoxProcType>(ty))
      return true;
    if (auto funcTy = mlir::dyn_cast<mlir::FunctionType>(ty)) {
      for (auto t : funcTy.getInputs())
        if (needsConversion(t))
          return true;
      for (auto t : funcTy.getResults())
        if (needsConversion(t))
          return true;
      return false;
    }
    if (auto tupleTy = mlir::dyn_cast<mlir::TupleType>(ty)) {
      for (auto t : tupleTy.getTypes())
        if (needsConversion(t))
          return true;
      return false;
    }
    if (auto recTy = mlir::dyn_cast<RecordType>(ty)) {
      auto [visited, inserted] = visitedTypes.try_emplace(ty, false);
      if (!inserted)
        return visited->second;
      bool wasAlreadyVisitingRecordType = needConversionIsVisitingRecordType;
      needConversionIsVisitingRecordType = true;
      bool result = false;
      for (auto t : recTy.getTypeList()) {
        if (needsConversion(t.second)) {
          result = true;
          break;
        }
      }
      // Only keep the result cached if the fir.type visited was a "top-level
      // type". Nested types with a recursive reference to the "top-level type"
      // may incorrectly have been resolved as not needed conversions because it
      // had not been determined yet if the "top-level type" needed conversion.
      // This is not an issue to determine the "top-level type" need of
      // conversion, but the result should not be kept and later used in other
      // contexts.
      needConversionIsVisitingRecordType = wasAlreadyVisitingRecordType;
      if (needConversionIsVisitingRecordType)
        visitedTypes.erase(ty);
      else
        visitedTypes.find(ty)->second = result;
      return result;
    }
    if (auto boxTy = mlir::dyn_cast<BaseBoxType>(ty))
      return needsConversion(boxTy.getEleTy());
    if (isa_ref_type(ty))
      return needsConversion(unwrapRefType(ty));
    if (auto t = mlir::dyn_cast<SequenceType>(ty))
      return needsConversion(unwrapSequenceType(ty));
    if (auto t = mlir::dyn_cast<TypeDescType>(ty))
      return needsConversion(t.getOfTy());
    return false;
  }

  BoxprocTypeRewriter(mlir::Location location) : loc{location} {
    addConversion([](mlir::Type ty) { return ty; });
    addConversion(
        [&](BoxProcType boxproc) { return convertType(boxproc.getEleTy()); });
    addConversion([&](mlir::TupleType tupTy) {
      llvm::SmallVector<mlir::Type> memTys;
      for (auto ty : tupTy.getTypes())
        memTys.push_back(convertType(ty));
      return mlir::TupleType::get(tupTy.getContext(), memTys);
    });
    addConversion([&](mlir::FunctionType funcTy) {
      llvm::SmallVector<mlir::Type> inTys;
      llvm::SmallVector<mlir::Type> resTys;
      for (auto ty : funcTy.getInputs())
        inTys.push_back(convertType(ty));
      for (auto ty : funcTy.getResults())
        resTys.push_back(convertType(ty));
      return mlir::FunctionType::get(funcTy.getContext(), inTys, resTys);
    });
    addConversion([&](ReferenceType ty) {
      return ReferenceType::get(convertType(ty.getEleTy()));
    });
    addConversion([&](PointerType ty) {
      return PointerType::get(convertType(ty.getEleTy()));
    });
    addConversion(
        [&](HeapType ty) { return HeapType::get(convertType(ty.getEleTy())); });
    addConversion([&](fir::LLVMPointerType ty) {
      return fir::LLVMPointerType::get(convertType(ty.getEleTy()));
    });
    addConversion(
        [&](BoxType ty) { return BoxType::get(convertType(ty.getEleTy())); });
    addConversion([&](ClassType ty) {
      return ClassType::get(convertType(ty.getEleTy()));
    });
    addConversion([&](SequenceType ty) {
      // TODO: add ty.getLayoutMap() as needed.
      return SequenceType::get(ty.getShape(), convertType(ty.getEleTy()));
    });
    addConversion([&](RecordType ty) -> mlir::Type {
      if (!needsConversion(ty))
        return ty;
      if (auto converted = convertedTypes.lookup(ty))
        return converted;
      auto rec = RecordType::get(ty.getContext(),
                                 ty.getName().str() + boxprocSuffix.str());
      if (rec.isFinalized())
        return rec;
      [[maybe_unused]] auto it = convertedTypes.try_emplace(ty, rec);
      assert(it.second && "expected ty to not be in the map");
      std::vector<RecordType::TypePair> ps = ty.getLenParamList();
      std::vector<RecordType::TypePair> cs;
      for (auto t : ty.getTypeList()) {
        if (needsConversion(t.second))
          cs.emplace_back(t.first, convertType(t.second));
        else
          cs.emplace_back(t.first, t.second);
      }
      rec.finalize(ps, cs);
      rec.pack(ty.isPacked());
      return rec;
    });
    addConversion([&](TypeDescType ty) {
      return TypeDescType::get(convertType(ty.getOfTy()));
    });
    addSourceMaterialization(materializeProcedure);
    addTargetMaterialization(materializeProcedure);
  }

  static mlir::Value materializeProcedure(mlir::OpBuilder &builder,
                                          BoxProcType type,
                                          mlir::ValueRange inputs,
                                          mlir::Location loc) {
    assert(inputs.size() == 1);
    return builder.create<ConvertOp>(loc, unwrapRefType(type.getEleTy()),
                                     inputs[0]);
  }

  void setLocation(mlir::Location location) { loc = location; }

private:
  // Maps to deal with recursive derived types (avoid infinite loops).
  // Caching is also beneficial for apps with big types (dozens of
  // components and or parent types), so the lifetime of the cache
  // is the whole pass.
  llvm::DenseMap<mlir::Type, bool> visitedTypes;
  bool needConversionIsVisitingRecordType = false;
  llvm::DenseMap<mlir::Type, mlir::Type> convertedTypes;
  mlir::Location loc;
};

/// A `boxproc` is an abstraction for a Fortran procedure reference. Typically,
/// Fortran procedures can be referenced directly through a function pointer.
/// However, Fortran has one-level dynamic scoping between a host procedure and
/// its internal procedures. This allows internal procedures to directly access
/// and modify the state of the host procedure's variables.
///
/// There are any number of possible implementations possible.
///
/// The implementation used here is to convert `boxproc` values to function
/// pointers everywhere. If a `boxproc` value includes a frame pointer to the
/// host procedure's data, then a thunk will be created at runtime to capture
/// the frame pointer during execution. In LLVM IR, the frame pointer is
/// designated with the `nest` attribute. The thunk's address will then be used
/// as the call target instead of the original function's address directly.
class BoxedProcedurePass
    : public fir::impl::BoxedProcedurePassBase<BoxedProcedurePass> {
public:
  using BoxedProcedurePassBase<BoxedProcedurePass>::BoxedProcedurePassBase;

  inline mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    if (options.useThunks) {
      auto *context = &getContext();
      mlir::IRRewriter rewriter(context);
      BoxprocTypeRewriter typeConverter(mlir::UnknownLoc::get(context));
      getModule().walk([&](mlir::Operation *op) {
        bool opIsValid = true;
        typeConverter.setLocation(op->getLoc());
        if (auto addr = mlir::dyn_cast<BoxAddrOp>(op)) {
          mlir::Type ty = addr.getVal().getType();
          mlir::Type resTy = addr.getResult().getType();
          if (llvm::isa<mlir::FunctionType>(ty) ||
              llvm::isa<fir::BoxProcType>(ty)) {
            // Rewrite all `fir.box_addr` ops on values of type `!fir.boxproc`
            // or function type to be `fir.convert` ops.
            rewriter.setInsertionPoint(addr);
            rewriter.replaceOpWithNewOp<ConvertOp>(
                addr, typeConverter.convertType(addr.getType()), addr.getVal());
            opIsValid = false;
          } else if (typeConverter.needsConversion(resTy)) {
            rewriter.startOpModification(op);
            op->getResult(0).setType(typeConverter.convertType(resTy));
            rewriter.finalizeOpModification(op);
          }
        } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
          mlir::FunctionType ty = func.getFunctionType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.startOpModification(func);
            auto toTy =
                mlir::cast<mlir::FunctionType>(typeConverter.convertType(ty));
            if (!func.empty())
              for (auto e : llvm::enumerate(toTy.getInputs())) {
                unsigned i = e.index();
                auto &block = func.front();
                block.insertArgument(i, e.value(), func.getLoc());
                block.getArgument(i + 1).replaceAllUsesWith(
                    block.getArgument(i));
                block.eraseArgument(i + 1);
              }
            func.setType(toTy);
            rewriter.finalizeOpModification(func);
          }
        } else if (auto embox = mlir::dyn_cast<EmboxProcOp>(op)) {
          // Rewrite all `fir.emboxproc` ops to either `fir.convert` or a thunk
          // as required.
          mlir::Type toTy = typeConverter.convertType(
              mlir::cast<BoxProcType>(embox.getType()).getEleTy());
          rewriter.setInsertionPoint(embox);
          if (embox.getHost()) {
            // Create the thunk.
            auto module = embox->getParentOfType<mlir::ModuleOp>();
            FirOpBuilder builder(rewriter, module);
            const auto triple{fir::getTargetTriple(module)};
            auto loc = embox.getLoc();
            mlir::Type i8Ty = builder.getI8Type();
            mlir::Type i8Ptr = builder.getRefType(i8Ty);
            // For PPC32 and PPC64, the thunk is populated by a call to
            // __trampoline_setup, which is defined in
            // compiler-rt/lib/builtins/trampoline_setup.c and requires the
            // thunk size greater than 32 bytes.  For AArch64, RISCV and x86_64,
            // the thunk setup doesn't go through __trampoline_setup and fits in
            // 32 bytes.
            fir::SequenceType::Extent thunkSize = triple.getTrampolineSize();
            mlir::Type buffTy = SequenceType::get({thunkSize}, i8Ty);
            auto buffer = builder.create<AllocaOp>(loc, buffTy);
            mlir::Value closure =
                builder.createConvert(loc, i8Ptr, embox.getHost());
            mlir::Value tramp = builder.createConvert(loc, i8Ptr, buffer);
            mlir::Value func =
                builder.createConvert(loc, i8Ptr, embox.getFunc());
            builder.create<fir::CallOp>(
                loc, factory::getLlvmInitTrampoline(builder),
                llvm::ArrayRef<mlir::Value>{tramp, func, closure});
            auto adjustCall = builder.create<fir::CallOp>(
                loc, factory::getLlvmAdjustTrampoline(builder),
                llvm::ArrayRef<mlir::Value>{tramp});
            rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy,
                                                   adjustCall.getResult(0));
            opIsValid = false;
          } else {
            // Just forward the function as a pointer.
            rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy,
                                                   embox.getFunc());
            opIsValid = false;
          }
        } else if (auto global = mlir::dyn_cast<GlobalOp>(op)) {
          auto ty = global.getType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.startOpModification(global);
            auto toTy = typeConverter.convertType(ty);
            global.setType(toTy);
            rewriter.finalizeOpModification(global);
          }
        } else if (auto mem = mlir::dyn_cast<AllocaOp>(op)) {
          auto ty = mem.getType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.setInsertionPoint(mem);
            auto toTy = typeConverter.convertType(unwrapRefType(ty));
            bool isPinned = mem.getPinned();
            llvm::StringRef uniqName =
                mem.getUniqName().value_or(llvm::StringRef());
            llvm::StringRef bindcName =
                mem.getBindcName().value_or(llvm::StringRef());
            rewriter.replaceOpWithNewOp<AllocaOp>(
                mem, toTy, uniqName, bindcName, isPinned, mem.getTypeparams(),
                mem.getShape());
            opIsValid = false;
          }
        } else if (auto mem = mlir::dyn_cast<AllocMemOp>(op)) {
          auto ty = mem.getType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.setInsertionPoint(mem);
            auto toTy = typeConverter.convertType(unwrapRefType(ty));
            llvm::StringRef uniqName =
                mem.getUniqName().value_or(llvm::StringRef());
            llvm::StringRef bindcName =
                mem.getBindcName().value_or(llvm::StringRef());
            rewriter.replaceOpWithNewOp<AllocMemOp>(
                mem, toTy, uniqName, bindcName, mem.getTypeparams(),
                mem.getShape());
            opIsValid = false;
          }
        } else if (auto coor = mlir::dyn_cast<CoordinateOp>(op)) {
          auto ty = coor.getType();
          mlir::Type baseTy = coor.getBaseType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(baseTy)) {
            rewriter.setInsertionPoint(coor);
            auto toTy = typeConverter.convertType(ty);
            auto toBaseTy = typeConverter.convertType(baseTy);
            rewriter.replaceOpWithNewOp<CoordinateOp>(
                coor, toTy, coor.getRef(), coor.getCoor(), toBaseTy,
                coor.getFieldIndicesAttr());
            opIsValid = false;
          }
        } else if (auto index = mlir::dyn_cast<FieldIndexOp>(op)) {
          auto ty = index.getType();
          mlir::Type onTy = index.getOnType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(onTy)) {
            rewriter.setInsertionPoint(index);
            auto toTy = typeConverter.convertType(ty);
            auto toOnTy = typeConverter.convertType(onTy);
            rewriter.replaceOpWithNewOp<FieldIndexOp>(
                index, toTy, index.getFieldId(), toOnTy, index.getTypeparams());
            opIsValid = false;
          }
        } else if (auto index = mlir::dyn_cast<LenParamIndexOp>(op)) {
          auto ty = index.getType();
          mlir::Type onTy = index.getOnType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(onTy)) {
            rewriter.setInsertionPoint(index);
            auto toTy = typeConverter.convertType(ty);
            auto toOnTy = typeConverter.convertType(onTy);
            rewriter.replaceOpWithNewOp<LenParamIndexOp>(
                index, toTy, index.getFieldId(), toOnTy, index.getTypeparams());
            opIsValid = false;
          }
        } else {
          rewriter.startOpModification(op);
          // Convert the operands if needed
          for (auto i : llvm::enumerate(op->getResultTypes()))
            if (typeConverter.needsConversion(i.value())) {
              auto toTy = typeConverter.convertType(i.value());
              op->getResult(i.index()).setType(toTy);
            }

          // Convert the type attributes if needed
          for (const mlir::NamedAttribute &attr : op->getAttrDictionary())
            if (auto tyAttr = llvm::dyn_cast<mlir::TypeAttr>(attr.getValue()))
              if (typeConverter.needsConversion(tyAttr.getValue())) {
                auto toTy = typeConverter.convertType(tyAttr.getValue());
                op->setAttr(attr.getName(), mlir::TypeAttr::get(toTy));
              }
          rewriter.finalizeOpModification(op);
        }
        // Ensure block arguments are updated if needed.
        if (opIsValid && op->getNumRegions() != 0) {
          rewriter.startOpModification(op);
          for (mlir::Region &region : op->getRegions())
            for (mlir::Block &block : region.getBlocks())
              for (mlir::BlockArgument blockArg : block.getArguments())
                if (typeConverter.needsConversion(blockArg.getType())) {
                  mlir::Type toTy =
                      typeConverter.convertType(blockArg.getType());
                  blockArg.setType(toTy);
                }
          rewriter.finalizeOpModification(op);
        }
      });
    }
  }

private:
  BoxedProcedureOptions options;
};
} // namespace
