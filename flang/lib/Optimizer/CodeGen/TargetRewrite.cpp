//===-- TargetRewrite.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target rewrite: rewriting of ops to make target-specific lowerings manifest.
// LLVM expects different lowering idioms to be used for distinct target
// triples. These distinctions are handled by this pass.
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_TARGETREWRITEPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-target-rewrite"

namespace {

/// Fixups for updating a FuncOp's arguments and return values.
struct FixupTy {
  enum class Codes {
    ArgumentAsLoad,
    ArgumentType,
    CharPair,
    ReturnAsStore,
    ReturnType,
    Split,
    Trailing,
    TrailingCharProc
  };

  FixupTy(Codes code, std::size_t index, std::size_t second = 0)
      : code{code}, index{index}, second{second} {}
  FixupTy(Codes code, std::size_t index,
          std::function<void(mlir::func::FuncOp)> &&finalizer)
      : code{code}, index{index}, finalizer{finalizer} {}
  FixupTy(Codes code, std::size_t index, std::size_t second,
          std::function<void(mlir::func::FuncOp)> &&finalizer)
      : code{code}, index{index}, second{second}, finalizer{finalizer} {}

  Codes code;
  std::size_t index;
  std::size_t second{};
  std::optional<std::function<void(mlir::func::FuncOp)>> finalizer{};
}; // namespace

/// Target-specific rewriting of the FIR. This is a prerequisite pass to code
/// generation that traverses the FIR and modifies types and operations to a
/// form that is appropriate for the specific target. LLVM IR has specific
/// idioms that are used for distinct target processor and ABI combinations.
class TargetRewrite : public fir::impl::TargetRewritePassBase<TargetRewrite> {
public:
  TargetRewrite(const fir::TargetRewriteOptions &options) {
    noCharacterConversion = options.noCharacterConversion;
    noComplexConversion = options.noComplexConversion;
    noStructConversion = options.noStructConversion;
  }

  void runOnOperation() override final {
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);

    auto mod = getModule();
    if (!forcedTargetTriple.empty())
      fir::setTargetTriple(mod, forcedTargetTriple);

    // TargetRewrite will require querying the type storage sizes, if it was
    // not set already, create a DataLayoutSpec for the ModuleOp now.
    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetDataLayout(mod, /*allowDefaultLayout=*/true);
    if (!dl) {
      mlir::emitError(mod.getLoc(),
                      "module operation must carry a data layout attribute "
                      "to perform target ABI rewrites on FIR");
      signalPassFailure();
      return;
    }

    auto specifics =
        fir::CodeGenSpecifics::get(mod.getContext(), fir::getTargetTriple(mod),
                                   fir::getKindMapping(mod), *dl);

    setMembers(specifics.get(), &rewriter, &*dl);

    // We may need to call stacksave/stackrestore later, so
    // create the FuncOps beforehand.
    fir::FirOpBuilder builder(rewriter, mod);
    builder.setInsertionPointToStart(mod.getBody());
    stackSaveFn = fir::factory::getLlvmStackSave(builder);
    stackRestoreFn = fir::factory::getLlvmStackRestore(builder);

    // Perform type conversion on signatures and call sites.
    if (mlir::failed(convertTypes(mod))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in converting types to target abi");
      signalPassFailure();
    }

    // Convert ops in target-specific patterns.
    mod.walk([&](mlir::Operation *op) {
      if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
        if (!hasPortableSignature(call.getFunctionType(), op))
          convertCallOp(call);
      } else if (auto dispatch = mlir::dyn_cast<fir::DispatchOp>(op)) {
        if (!hasPortableSignature(dispatch.getFunctionType(), op))
          convertCallOp(dispatch);
      } else if (auto addr = mlir::dyn_cast<fir::AddrOfOp>(op)) {
        if (addr.getType().isa<mlir::FunctionType>() &&
            !hasPortableSignature(addr.getType(), op))
          convertAddrOp(addr);
      }
    });

    clearMembers();
  }

  mlir::ModuleOp getModule() { return getOperation(); }

  template <typename A, typename B, typename C>
  std::optional<std::function<mlir::Value(mlir::Operation *)>>
  rewriteCallComplexResultType(
      mlir::Location loc, A ty, B &newResTys,
      fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs, C &newOpers,
      mlir::Value &savedStackPtr) {
    if (noComplexConversion) {
      newResTys.push_back(ty);
      return std::nullopt;
    }
    auto m = specifics->complexReturnType(loc, ty.getElementType());
    // Currently targets mandate COMPLEX is a single aggregate or packed
    // scalar, including the sret case.
    assert(m.size() == 1 && "target of complex return not supported");
    auto resTy = std::get<mlir::Type>(m[0]);
    auto attr = std::get<fir::CodeGenSpecifics::Attributes>(m[0]);
    if (attr.isSRet()) {
      assert(fir::isa_ref_type(resTy) && "must be a memory reference type");
      // Save the stack pointer, if it has not been saved for this call yet.
      // We will need to restore it after the call, because the alloca
      // needs to be deallocated.
      if (!savedStackPtr)
        savedStackPtr = genStackSave(loc);
      mlir::Value stack =
          rewriter->create<fir::AllocaOp>(loc, fir::dyn_cast_ptrEleTy(resTy));
      newInTyAndAttrs.push_back(m[0]);
      newOpers.push_back(stack);
      return [=](mlir::Operation *) -> mlir::Value {
        auto memTy = fir::ReferenceType::get(ty);
        auto cast = rewriter->create<fir::ConvertOp>(loc, memTy, stack);
        return rewriter->create<fir::LoadOp>(loc, cast);
      };
    }
    newResTys.push_back(resTy);
    return [=, &savedStackPtr](mlir::Operation *call) -> mlir::Value {
      // We are going to generate an alloca, so save the stack pointer.
      if (!savedStackPtr)
        savedStackPtr = genStackSave(loc);
      auto mem = rewriter->create<fir::AllocaOp>(loc, resTy);
      rewriter->create<fir::StoreOp>(loc, call->getResult(0), mem);
      auto memTy = fir::ReferenceType::get(ty);
      auto cast = rewriter->create<fir::ConvertOp>(loc, memTy, mem);
      return rewriter->create<fir::LoadOp>(loc, cast);
    };
  }

  void passArgumentOnStackOrWithNewType(
      mlir::Location loc, fir::CodeGenSpecifics::TypeAndAttr newTypeAndAttr,
      mlir::Type oldType, mlir::Value oper,
      llvm::SmallVectorImpl<mlir::Value> &newOpers,
      mlir::Value &savedStackPtr) {
    auto resTy = std::get<mlir::Type>(newTypeAndAttr);
    auto attr = std::get<fir::CodeGenSpecifics::Attributes>(newTypeAndAttr);
    auto oldRefTy = fir::ReferenceType::get(oldType);
    // We are going to generate an alloca, so save the stack pointer.
    if (!savedStackPtr)
      savedStackPtr = genStackSave(loc);
    if (attr.isByVal()) {
      mlir::Value mem = rewriter->create<fir::AllocaOp>(loc, oldType);
      rewriter->create<fir::StoreOp>(loc, oper, mem);
      if (mem.getType() != resTy)
        mem = rewriter->create<fir::ConvertOp>(loc, resTy, mem);
      newOpers.push_back(mem);
    } else {
      auto mem = rewriter->create<fir::AllocaOp>(loc, resTy);
      auto cast = rewriter->create<fir::ConvertOp>(loc, oldRefTy, mem);
      rewriter->create<fir::StoreOp>(loc, oper, cast);
      newOpers.push_back(rewriter->create<fir::LoadOp>(loc, mem));
    }
  }

  template <typename CPLX>
  void rewriteCallComplexInputType(
      mlir::Location loc, CPLX ty, mlir::Value oper,
      fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
      llvm::SmallVectorImpl<mlir::Value> &newOpers,
      mlir::Value &savedStackPtr) {
    if (noComplexConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(ty));
      newOpers.push_back(oper);
      return;
    }

    auto m = specifics->complexArgumentType(loc, ty.getElementType());
    if (m.size() == 1) {
      // COMPLEX is a single aggregate
      passArgumentOnStackOrWithNewType(loc, m[0], ty, oper, newOpers,
                                       savedStackPtr);
      newInTyAndAttrs.push_back(m[0]);
    } else {
      assert(m.size() == 2);
      // COMPLEX is split into 2 separate arguments
      auto iTy = rewriter->getIntegerType(32);
      for (auto e : llvm::enumerate(m)) {
        auto &tup = e.value();
        auto ty = std::get<mlir::Type>(tup);
        auto index = e.index();
        auto idx = rewriter->getIntegerAttr(iTy, index);
        auto val = rewriter->create<fir::ExtractValueOp>(
            loc, ty, oper, rewriter->getArrayAttr(idx));
        newInTyAndAttrs.push_back(tup);
        newOpers.push_back(val);
      }
    }
  }

  void rewriteCallStructInputType(
      mlir::Location loc, fir::RecordType recTy, mlir::Value oper,
      fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
      llvm::SmallVectorImpl<mlir::Value> &newOpers,
      mlir::Value &savedStackPtr) {
    if (noStructConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(recTy));
      newOpers.push_back(oper);
      return;
    }
    auto structArgs =
        specifics->structArgumentType(loc, recTy, newInTyAndAttrs);
    if (structArgs.size() != 1)
      TODO(loc, "splitting BIND(C), VALUE derived type into several arguments");
    passArgumentOnStackOrWithNewType(loc, structArgs[0], recTy, oper, newOpers,
                                     savedStackPtr);
    structArgs.push_back(structArgs[0]);
  }

  static bool hasByValOrSRetArgs(
      const fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs) {
    return llvm::any_of(newInTyAndAttrs, [](auto arg) {
      const auto &attr = std::get<fir::CodeGenSpecifics::Attributes>(arg);
      return attr.isByVal() || attr.isSRet();
    });
  }

  // Convert fir.call and fir.dispatch Ops.
  template <typename A>
  void convertCallOp(A callOp) {
    auto fnTy = callOp.getFunctionType();
    auto loc = callOp.getLoc();
    rewriter->setInsertionPoint(callOp);
    llvm::SmallVector<mlir::Type> newResTys;
    fir::CodeGenSpecifics::Marshalling newInTyAndAttrs;
    llvm::SmallVector<mlir::Value> newOpers;
    mlir::Value savedStackPtr = nullptr;

    // If the call is indirect, the first argument must still be the function
    // to call.
    int dropFront = 0;
    if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
      if (!callOp.getCallee()) {
        newInTyAndAttrs.push_back(
            fir::CodeGenSpecifics::getTypeAndAttr(fnTy.getInput(0)));
        newOpers.push_back(callOp.getOperand(0));
        dropFront = 1;
      }
    } else {
      dropFront = 1; // First operand is the polymorphic object.
    }

    // Determine the rewrite function, `wrap`, for the result value.
    std::optional<std::function<mlir::Value(mlir::Operation *)>> wrap;
    if (fnTy.getResults().size() == 1) {
      mlir::Type ty = fnTy.getResult(0);
      llvm::TypeSwitch<mlir::Type>(ty)
          .template Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            wrap = rewriteCallComplexResultType(loc, cmplx, newResTys,
                                                newInTyAndAttrs, newOpers,
                                                savedStackPtr);
          })
          .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            wrap = rewriteCallComplexResultType(loc, cmplx, newResTys,
                                                newInTyAndAttrs, newOpers,
                                                savedStackPtr);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });
    } else if (fnTy.getResults().size() > 1) {
      TODO(loc, "multiple results not supported yet");
    }

    llvm::SmallVector<mlir::Type> trailingInTys;
    llvm::SmallVector<mlir::Value> trailingOpers;
    unsigned passArgShift = 0;
    for (auto e : llvm::enumerate(
             llvm::zip(fnTy.getInputs().drop_front(dropFront),
                       callOp.getOperands().drop_front(dropFront)))) {
      mlir::Type ty = std::get<0>(e.value());
      mlir::Value oper = std::get<1>(e.value());
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .template Case<fir::BoxCharType>([&](fir::BoxCharType boxTy) {
            bool sret;
            if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
              if (noCharacterConversion) {
                newInTyAndAttrs.push_back(
                    fir::CodeGenSpecifics::getTypeAndAttr(boxTy));
                newOpers.push_back(oper);
                return;
              }
              sret = callOp.getCallee() &&
                     functionArgIsSRet(
                         index, getModule().lookupSymbol<mlir::func::FuncOp>(
                                    *callOp.getCallee()));
            } else {
              // TODO: dispatch case; how do we put arguments on a call?
              // We cannot put both an sret and the dispatch object first.
              sret = false;
              TODO(loc, "dispatch + sret not supported yet");
            }
            auto m = specifics->boxcharArgumentType(boxTy.getEleTy(), sret);
            auto unbox = rewriter->create<fir::UnboxCharOp>(
                loc, std::get<mlir::Type>(m[0]), std::get<mlir::Type>(m[1]),
                oper);
            // unboxed CHARACTER arguments
            for (auto e : llvm::enumerate(m)) {
              unsigned idx = e.index();
              auto attr =
                  std::get<fir::CodeGenSpecifics::Attributes>(e.value());
              auto argTy = std::get<mlir::Type>(e.value());
              if (attr.isAppend()) {
                trailingInTys.push_back(argTy);
                trailingOpers.push_back(unbox.getResult(idx));
              } else {
                newInTyAndAttrs.push_back(e.value());
                newOpers.push_back(unbox.getResult(idx));
              }
            }
          })
          .template Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            rewriteCallComplexInputType(loc, cmplx, oper, newInTyAndAttrs,
                                        newOpers, savedStackPtr);
          })
          .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            rewriteCallComplexInputType(loc, cmplx, oper, newInTyAndAttrs,
                                        newOpers, savedStackPtr);
          })
          .template Case<fir::RecordType>([&](fir::RecordType recTy) {
            rewriteCallStructInputType(loc, recTy, oper, newInTyAndAttrs,
                                       newOpers, savedStackPtr);
          })
          .template Case<mlir::TupleType>([&](mlir::TupleType tuple) {
            if (fir::isCharacterProcedureTuple(tuple)) {
              mlir::ModuleOp module = getModule();
              if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
                if (callOp.getCallee()) {
                  llvm::StringRef charProcAttr =
                      fir::getCharacterProcedureDummyAttrName();
                  // The charProcAttr attribute is only used as a safety to
                  // confirm that this is a dummy procedure and should be split.
                  // It cannot be used to match because attributes are not
                  // available in case of indirect calls.
                  auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(
                      *callOp.getCallee());
                  if (funcOp &&
                      !funcOp.template getArgAttrOfType<mlir::UnitAttr>(
                          index, charProcAttr))
                    mlir::emitError(loc, "tuple argument will be split even "
                                         "though it does not have the `" +
                                             charProcAttr + "` attribute");
                }
              }
              mlir::Type funcPointerType = tuple.getType(0);
              mlir::Type lenType = tuple.getType(1);
              fir::FirOpBuilder builder(*rewriter, module);
              auto [funcPointer, len] =
                  fir::factory::extractCharacterProcedureTuple(builder, loc,
                                                               oper);
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(funcPointerType));
              newOpers.push_back(funcPointer);
              trailingInTys.push_back(lenType);
              trailingOpers.push_back(len);
            } else {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(tuple));
              newOpers.push_back(oper);
            }
          })
          .Default([&](mlir::Type ty) {
            if constexpr (std::is_same_v<std::decay_t<A>, fir::DispatchOp>) {
              if (callOp.getPassArgPos() && *callOp.getPassArgPos() == index)
                passArgShift = newOpers.size() - *callOp.getPassArgPos();
            }
            newInTyAndAttrs.push_back(
                fir::CodeGenSpecifics::getTypeAndAttr(ty));
            newOpers.push_back(oper);
          });
    }

    llvm::SmallVector<mlir::Type> newInTypes = toTypeList(newInTyAndAttrs);
    newInTypes.insert(newInTypes.end(), trailingInTys.begin(),
                      trailingInTys.end());
    newOpers.insert(newOpers.end(), trailingOpers.begin(), trailingOpers.end());

    llvm::SmallVector<mlir::Value, 1> newCallResults;
    if constexpr (std::is_same_v<std::decay_t<A>, fir::CallOp>) {
      fir::CallOp newCall;
      if (callOp.getCallee()) {
        newCall =
            rewriter->create<A>(loc, *callOp.getCallee(), newResTys, newOpers);
      } else {
        // TODO: llvm dialect must be updated to propagate argument on
        // attributes for indirect calls. See:
        // https://discourse.llvm.org/t/should-llvm-callop-be-able-to-carry-argument-attributes-for-indirect-calls/75431
        if (hasByValOrSRetArgs(newInTyAndAttrs))
          TODO(loc,
               "passing argument or result on the stack in indirect calls");
        newOpers[0].setType(mlir::FunctionType::get(
            callOp.getContext(),
            mlir::TypeRange{newInTypes}.drop_front(dropFront), newResTys));
        newCall = rewriter->create<A>(loc, newResTys, newOpers);
      }
      LLVM_DEBUG(llvm::dbgs() << "replacing call with " << newCall << '\n');
      if (wrap)
        newCallResults.push_back((*wrap)(newCall.getOperation()));
      else
        newCallResults.append(newCall.result_begin(), newCall.result_end());
    } else {
      fir::DispatchOp dispatchOp = rewriter->create<A>(
          loc, newResTys, rewriter->getStringAttr(callOp.getMethod()),
          callOp.getOperands()[0], newOpers,
          rewriter->getI32IntegerAttr(*callOp.getPassArgPos() + passArgShift));
      if (wrap)
        newCallResults.push_back((*wrap)(dispatchOp.getOperation()));
      else
        newCallResults.append(dispatchOp.result_begin(),
                              dispatchOp.result_end());
    }

    if (newCallResults.size() <= 1) {
      if (savedStackPtr) {
        if (newCallResults.size() == 1) {
          // We assume that all the allocas are inserted before
          // the operation that defines the new call result.
          rewriter->setInsertionPointAfterValue(newCallResults[0]);
        } else {
          // If the call does not have results, then insert
          // stack restore after the original call operation.
          rewriter->setInsertionPointAfter(callOp);
        }
        genStackRestore(loc, savedStackPtr);
      }
      replaceOp(callOp, newCallResults);
    } else {
      // The TODO is duplicated here to make sure this part
      // handles the stackrestore insertion properly, if
      // we add support for multiple call results.
      TODO(loc, "multiple results not supported yet");
    }
  }

  // Result type fixup for fir::ComplexType and mlir::ComplexType
  template <typename A, typename B>
  void lowerComplexSignatureRes(
      mlir::Location loc, A cmplx, B &newResTys,
      fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs) {
    if (noComplexConversion) {
      newResTys.push_back(cmplx);
      return;
    }
    for (auto &tup :
         specifics->complexReturnType(loc, cmplx.getElementType())) {
      auto argTy = std::get<mlir::Type>(tup);
      if (std::get<fir::CodeGenSpecifics::Attributes>(tup).isSRet())
        newInTyAndAttrs.push_back(tup);
      else
        newResTys.push_back(argTy);
    }
  }

  // Argument type fixup for fir::ComplexType and mlir::ComplexType
  template <typename A>
  void lowerComplexSignatureArg(
      mlir::Location loc, A cmplx,
      fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs) {
    if (noComplexConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(cmplx));
    } else {
      auto cplxArgs =
          specifics->complexArgumentType(loc, cmplx.getElementType());
      newInTyAndAttrs.insert(newInTyAndAttrs.end(), cplxArgs.begin(),
                             cplxArgs.end());
    }
  }

  void
  lowerStructSignatureArg(mlir::Location loc, fir::RecordType recTy,
                          fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs) {
    if (noStructConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(recTy));
      return;
    }
    auto structArgs =
        specifics->structArgumentType(loc, recTy, newInTyAndAttrs);
    newInTyAndAttrs.insert(newInTyAndAttrs.end(), structArgs.begin(),
                           structArgs.end());
  }

  llvm::SmallVector<mlir::Type>
  toTypeList(const fir::CodeGenSpecifics::Marshalling &marshalled) {
    llvm::SmallVector<mlir::Type> typeList;
    for (auto &typeAndAttr : marshalled)
      typeList.emplace_back(std::get<mlir::Type>(typeAndAttr));
    return typeList;
  }

  /// Taking the address of a function. Modify the signature as needed.
  void convertAddrOp(fir::AddrOfOp addrOp) {
    rewriter->setInsertionPoint(addrOp);
    auto addrTy = addrOp.getType().cast<mlir::FunctionType>();
    fir::CodeGenSpecifics::Marshalling newInTyAndAttrs;
    llvm::SmallVector<mlir::Type> newResTys;
    auto loc = addrOp.getLoc();
    for (mlir::Type ty : addrTy.getResults()) {
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::ComplexType>([&](fir::ComplexType ty) {
            lowerComplexSignatureRes(loc, ty, newResTys, newInTyAndAttrs);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
            lowerComplexSignatureRes(loc, ty, newResTys, newInTyAndAttrs);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });
    }
    llvm::SmallVector<mlir::Type> trailingInTys;
    for (mlir::Type ty : addrTy.getInputs()) {
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::BoxCharType>([&](auto box) {
            if (noCharacterConversion) {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(box));
            } else {
              for (auto &tup : specifics->boxcharArgumentType(box.getEleTy())) {
                auto attr = std::get<fir::CodeGenSpecifics::Attributes>(tup);
                auto argTy = std::get<mlir::Type>(tup);
                if (attr.isAppend())
                  trailingInTys.push_back(argTy);
                else
                  newInTyAndAttrs.push_back(tup);
              }
            }
          })
          .Case<fir::ComplexType>([&](fir::ComplexType ty) {
            lowerComplexSignatureArg(loc, ty, newInTyAndAttrs);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType ty) {
            lowerComplexSignatureArg(loc, ty, newInTyAndAttrs);
          })
          .Case<mlir::TupleType>([&](mlir::TupleType tuple) {
            if (fir::isCharacterProcedureTuple(tuple)) {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(tuple.getType(0)));
              trailingInTys.push_back(tuple.getType(1));
            } else {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(ty));
            }
          })
          .template Case<fir::RecordType>([&](fir::RecordType recTy) {
            lowerStructSignatureArg(loc, recTy, newInTyAndAttrs);
          })
          .Default([&](mlir::Type ty) {
            newInTyAndAttrs.push_back(
                fir::CodeGenSpecifics::getTypeAndAttr(ty));
          });
    }
    llvm::SmallVector<mlir::Type> newInTypes = toTypeList(newInTyAndAttrs);
    // append trailing input types
    newInTypes.insert(newInTypes.end(), trailingInTys.begin(),
                      trailingInTys.end());
    // replace this op with a new one with the updated signature
    auto newTy = rewriter->getFunctionType(newInTypes, newResTys);
    auto newOp = rewriter->create<fir::AddrOfOp>(addrOp.getLoc(), newTy,
                                                 addrOp.getSymbol());
    replaceOp(addrOp, newOp.getResult());
  }

  /// Convert the type signatures on all the functions present in the module.
  /// As the type signature is being changed, this must also update the
  /// function itself to use any new arguments, etc.
  mlir::LogicalResult convertTypes(mlir::ModuleOp mod) {
    for (auto fn : mod.getOps<mlir::func::FuncOp>())
      convertSignature(fn);
    return mlir::success();
  }

  // Returns true if the function should be interoperable with C.
  static bool isFuncWithCCallingConvention(mlir::Operation *op) {
    auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!funcOp)
      return false;
    return op->hasAttrOfType<mlir::UnitAttr>(
               fir::FIROpsDialect::getFirRuntimeAttrName()) ||
           op->hasAttrOfType<mlir::StringAttr>(fir::getSymbolAttrName());
  }

  /// If the signature does not need any special target-specific conversions,
  /// then it is considered portable for any target, and this function will
  /// return `true`. Otherwise, the signature is not portable and `false` is
  /// returned.
  bool hasPortableSignature(mlir::Type signature, mlir::Operation *op) {
    assert(signature.isa<mlir::FunctionType>());
    auto func = signature.dyn_cast<mlir::FunctionType>();
    bool hasCCallingConv = isFuncWithCCallingConvention(op);
    for (auto ty : func.getResults())
      if ((ty.isa<fir::BoxCharType>() && !noCharacterConversion) ||
          (fir::isa_complex(ty) && !noComplexConversion) ||
          (ty.isa<mlir::IntegerType>() && hasCCallingConv)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    for (auto ty : func.getInputs())
      if (((ty.isa<fir::BoxCharType>() || fir::isCharacterProcedureTuple(ty)) &&
           !noCharacterConversion) ||
          (fir::isa_complex(ty) && !noComplexConversion) ||
          (ty.isa<mlir::IntegerType>() && hasCCallingConv) ||
          (ty.isa<fir::RecordType>() && !noStructConversion)) {
        LLVM_DEBUG(llvm::dbgs() << "rewrite " << signature << " for target\n");
        return false;
      }
    return true;
  }

  /// Determine if the signature has host associations. The host association
  /// argument may need special target specific rewriting.
  static bool hasHostAssociations(mlir::func::FuncOp func) {
    std::size_t end = func.getFunctionType().getInputs().size();
    for (std::size_t i = 0; i < end; ++i)
      if (func.getArgAttrOfType<mlir::UnitAttr>(i, fir::getHostAssocAttrName()))
        return true;
    return false;
  }

  /// Rewrite the signatures and body of the `FuncOp`s in the module for
  /// the immediately subsequent target code gen.
  void convertSignature(mlir::func::FuncOp func) {
    auto funcTy = func.getFunctionType().cast<mlir::FunctionType>();
    if (hasPortableSignature(funcTy, func) && !hasHostAssociations(func))
      return;
    llvm::SmallVector<mlir::Type> newResTys;
    fir::CodeGenSpecifics::Marshalling newInTyAndAttrs;
    llvm::SmallVector<std::pair<unsigned, mlir::NamedAttribute>> savedAttrs;
    llvm::SmallVector<std::pair<unsigned, mlir::NamedAttribute>> extraAttrs;
    llvm::SmallVector<FixupTy> fixups;
    llvm::SmallVector<std::pair<unsigned, mlir::NamedAttrList>, 1> resultAttrs;

    // Save argument attributes in case there is a shift so we can replace them
    // correctly.
    for (auto e : llvm::enumerate(funcTy.getInputs())) {
      unsigned index = e.index();
      llvm::ArrayRef<mlir::NamedAttribute> attrs =
          mlir::function_interface_impl::getArgAttrs(func, index);
      for (mlir::NamedAttribute attr : attrs) {
        savedAttrs.push_back({index, attr});
      }
    }

    // Convert return value(s)
    for (auto ty : funcTy.getResults())
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            if (noComplexConversion)
              newResTys.push_back(cmplx);
            else
              doComplexReturn(func, cmplx, newResTys, newInTyAndAttrs, fixups);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            if (noComplexConversion)
              newResTys.push_back(cmplx);
            else
              doComplexReturn(func, cmplx, newResTys, newInTyAndAttrs, fixups);
          })
          .Case<mlir::IntegerType>([&](mlir::IntegerType intTy) {
            auto m = specifics->integerArgumentType(func.getLoc(), intTy);
            assert(m.size() == 1);
            auto attr = std::get<fir::CodeGenSpecifics::Attributes>(m[0]);
            auto retTy = std::get<mlir::Type>(m[0]);
            std::size_t resId = newResTys.size();
            llvm::StringRef extensionAttrName = attr.getIntExtensionAttrName();
            if (!extensionAttrName.empty() &&
                isFuncWithCCallingConvention(func))
              resultAttrs.emplace_back(
                  resId, rewriter->getNamedAttr(extensionAttrName,
                                                rewriter->getUnitAttr()));
            newResTys.push_back(retTy);
          })
          .Default([&](mlir::Type ty) { newResTys.push_back(ty); });

    // Saved potential shift in argument. Handling of result can add arguments
    // at the beginning of the function signature.
    unsigned argumentShift = newInTyAndAttrs.size();

    // Convert arguments
    llvm::SmallVector<mlir::Type> trailingTys;
    for (auto e : llvm::enumerate(funcTy.getInputs())) {
      auto ty = e.value();
      unsigned index = e.index();
      llvm::TypeSwitch<mlir::Type>(ty)
          .Case<fir::BoxCharType>([&](fir::BoxCharType boxTy) {
            if (noCharacterConversion) {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(boxTy));
            } else {
              // Convert a CHARACTER argument type. This can involve separating
              // the pointer and the LEN into two arguments and moving the LEN
              // argument to the end of the arg list.
              bool sret = functionArgIsSRet(index, func);
              for (auto e : llvm::enumerate(specifics->boxcharArgumentType(
                       boxTy.getEleTy(), sret))) {
                auto &tup = e.value();
                auto index = e.index();
                auto attr = std::get<fir::CodeGenSpecifics::Attributes>(tup);
                auto argTy = std::get<mlir::Type>(tup);
                if (attr.isAppend()) {
                  trailingTys.push_back(argTy);
                } else {
                  if (sret) {
                    fixups.emplace_back(FixupTy::Codes::CharPair,
                                        newInTyAndAttrs.size(), index);
                  } else {
                    fixups.emplace_back(FixupTy::Codes::Trailing,
                                        newInTyAndAttrs.size(),
                                        trailingTys.size());
                  }
                  newInTyAndAttrs.push_back(tup);
                }
              }
            }
          })
          .Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
            doComplexArg(func, cmplx, newInTyAndAttrs, fixups);
          })
          .Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
            doComplexArg(func, cmplx, newInTyAndAttrs, fixups);
          })
          .Case<mlir::TupleType>([&](mlir::TupleType tuple) {
            if (fir::isCharacterProcedureTuple(tuple)) {
              fixups.emplace_back(FixupTy::Codes::TrailingCharProc,
                                  newInTyAndAttrs.size(), trailingTys.size());
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(tuple.getType(0)));
              trailingTys.push_back(tuple.getType(1));
            } else {
              newInTyAndAttrs.push_back(
                  fir::CodeGenSpecifics::getTypeAndAttr(ty));
            }
          })
          .Case<mlir::IntegerType>([&](mlir::IntegerType intTy) {
            auto m = specifics->integerArgumentType(func.getLoc(), intTy);
            assert(m.size() == 1);
            auto attr = std::get<fir::CodeGenSpecifics::Attributes>(m[0]);
            auto argNo = newInTyAndAttrs.size();
            llvm::StringRef extensionAttrName = attr.getIntExtensionAttrName();
            if (!extensionAttrName.empty() &&
                isFuncWithCCallingConvention(func))
              fixups.emplace_back(FixupTy::Codes::ArgumentType, argNo,
                                  [=](mlir::func::FuncOp func) {
                                    func.setArgAttr(
                                        argNo, extensionAttrName,
                                        mlir::UnitAttr::get(func.getContext()));
                                  });

            newInTyAndAttrs.push_back(m[0]);
          })
          .template Case<fir::RecordType>([&](fir::RecordType recTy) {
            doStructArg(func, recTy, newInTyAndAttrs, fixups);
          })
          .Default([&](mlir::Type ty) {
            newInTyAndAttrs.push_back(
                fir::CodeGenSpecifics::getTypeAndAttr(ty));
          });

      if (func.getArgAttrOfType<mlir::UnitAttr>(index,
                                                fir::getHostAssocAttrName())) {
        extraAttrs.push_back(
            {newInTyAndAttrs.size() - 1,
             rewriter->getNamedAttr("llvm.nest", rewriter->getUnitAttr())});
      }
    }

    if (!func.empty()) {
      // If the function has a body, then apply the fixups to the arguments and
      // return ops as required. These fixups are done in place.
      auto loc = func.getLoc();
      const auto fixupSize = fixups.size();
      const auto oldArgTys = func.getFunctionType().getInputs();
      int offset = 0;
      for (std::remove_const_t<decltype(fixupSize)> i = 0; i < fixupSize; ++i) {
        const auto &fixup = fixups[i];
        mlir::Type fixupType =
            fixup.index < newInTyAndAttrs.size()
                ? std::get<mlir::Type>(newInTyAndAttrs[fixup.index])
                : mlir::Type{};
        switch (fixup.code) {
        case FixupTy::Codes::ArgumentAsLoad: {
          // Argument was pass-by-value, but is now pass-by-reference and
          // possibly with a different element type.
          auto newArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          rewriter->setInsertionPointToStart(&func.front());
          auto oldArgTy =
              fir::ReferenceType::get(oldArgTys[fixup.index - offset]);
          auto cast = rewriter->create<fir::ConvertOp>(loc, oldArgTy, newArg);
          auto load = rewriter->create<fir::LoadOp>(loc, cast);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(load);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        case FixupTy::Codes::ArgumentType: {
          // Argument is pass-by-value, but its type has likely been modified to
          // suit the target ABI convention.
          auto oldArgTy =
              fir::ReferenceType::get(oldArgTys[fixup.index - offset]);
          // If type did not change, keep the original argument.
          if (fixupType == oldArgTy)
            break;

          auto newArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          rewriter->setInsertionPointToStart(&func.front());
          auto mem = rewriter->create<fir::AllocaOp>(loc, fixupType);
          rewriter->create<fir::StoreOp>(loc, newArg, mem);
          auto cast = rewriter->create<fir::ConvertOp>(loc, oldArgTy, mem);
          mlir::Value load = rewriter->create<fir::LoadOp>(loc, cast);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(load);
          func.front().eraseArgument(fixup.index + 1);
          LLVM_DEBUG(llvm::dbgs()
                     << "old argument: " << oldArgTy.getEleTy()
                     << ", repl: " << load << ", new argument: "
                     << func.getArgument(fixup.index).getType() << '\n');
        } break;
        case FixupTy::Codes::CharPair: {
          // The FIR boxchar argument has been split into a pair of distinct
          // arguments that are in juxtaposition to each other.
          auto newArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          if (fixup.second == 1) {
            rewriter->setInsertionPointToStart(&func.front());
            auto boxTy = oldArgTys[fixup.index - offset - fixup.second];
            auto box = rewriter->create<fir::EmboxCharOp>(
                loc, boxTy, func.front().getArgument(fixup.index - 1), newArg);
            func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
            func.front().eraseArgument(fixup.index + 1);
            offset++;
          }
        } break;
        case FixupTy::Codes::ReturnAsStore: {
          // The value being returned is now being returned in memory (callee
          // stack space) through a hidden reference argument.
          auto newArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          offset++;
          func.walk([&](mlir::func::ReturnOp ret) {
            rewriter->setInsertionPoint(ret);
            auto oldOper = ret.getOperand(0);
            auto oldOperTy = fir::ReferenceType::get(oldOper.getType());
            auto cast =
                rewriter->create<fir::ConvertOp>(loc, oldOperTy, newArg);
            rewriter->create<fir::StoreOp>(loc, oldOper, cast);
            rewriter->create<mlir::func::ReturnOp>(loc);
            ret.erase();
          });
        } break;
        case FixupTy::Codes::ReturnType: {
          // The function is still returning a value, but its type has likely
          // changed to suit the target ABI convention.
          func.walk([&](mlir::func::ReturnOp ret) {
            rewriter->setInsertionPoint(ret);
            auto oldOper = ret.getOperand(0);
            auto oldOperTy = fir::ReferenceType::get(oldOper.getType());
            auto mem =
                rewriter->create<fir::AllocaOp>(loc, newResTys[fixup.index]);
            auto cast = rewriter->create<fir::ConvertOp>(loc, oldOperTy, mem);
            rewriter->create<fir::StoreOp>(loc, oldOper, cast);
            mlir::Value load = rewriter->create<fir::LoadOp>(loc, mem);
            rewriter->create<mlir::func::ReturnOp>(loc, load);
            ret.erase();
          });
        } break;
        case FixupTy::Codes::Split: {
          // The FIR argument has been split into a pair of distinct arguments
          // that are in juxtaposition to each other. (For COMPLEX value.)
          auto newArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          if (fixup.second == 1) {
            rewriter->setInsertionPointToStart(&func.front());
            auto cplxTy = oldArgTys[fixup.index - offset - fixup.second];
            auto undef = rewriter->create<fir::UndefOp>(loc, cplxTy);
            auto iTy = rewriter->getIntegerType(32);
            auto zero = rewriter->getIntegerAttr(iTy, 0);
            auto one = rewriter->getIntegerAttr(iTy, 1);
            auto cplx1 = rewriter->create<fir::InsertValueOp>(
                loc, cplxTy, undef, func.front().getArgument(fixup.index - 1),
                rewriter->getArrayAttr(zero));
            auto cplx = rewriter->create<fir::InsertValueOp>(
                loc, cplxTy, cplx1, newArg, rewriter->getArrayAttr(one));
            func.getArgument(fixup.index + 1).replaceAllUsesWith(cplx);
            func.front().eraseArgument(fixup.index + 1);
            offset++;
          }
        } break;
        case FixupTy::Codes::Trailing: {
          // The FIR argument has been split into a pair of distinct arguments.
          // The first part of the pair appears in the original argument
          // position. The second part of the pair is appended after all the
          // original arguments. (Boxchar arguments.)
          auto newBufArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          auto newLenArg =
              func.front().addArgument(trailingTys[fixup.second], loc);
          auto boxTy = oldArgTys[fixup.index - offset];
          rewriter->setInsertionPointToStart(&func.front());
          auto box = rewriter->create<fir::EmboxCharOp>(loc, boxTy, newBufArg,
                                                        newLenArg);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(box);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        case FixupTy::Codes::TrailingCharProc: {
          // The FIR character procedure argument tuple must be split into a
          // pair of distinct arguments. The first part of the pair appears in
          // the original argument position. The second part of the pair is
          // appended after all the original arguments.
          auto newProcPointerArg =
              func.front().insertArgument(fixup.index, fixupType, loc);
          auto newLenArg =
              func.front().addArgument(trailingTys[fixup.second], loc);
          auto tupleType = oldArgTys[fixup.index - offset];
          rewriter->setInsertionPointToStart(&func.front());
          fir::FirOpBuilder builder(*rewriter, getModule());
          auto tuple = fir::factory::createCharacterProcedureTuple(
              builder, loc, tupleType, newProcPointerArg, newLenArg);
          func.getArgument(fixup.index + 1).replaceAllUsesWith(tuple);
          func.front().eraseArgument(fixup.index + 1);
        } break;
        }
      }
    }

    llvm::SmallVector<mlir::Type> newInTypes = toTypeList(newInTyAndAttrs);
    // Set the new type and finalize the arguments, etc.
    newInTypes.insert(newInTypes.end(), trailingTys.begin(), trailingTys.end());
    auto newFuncTy =
        mlir::FunctionType::get(func.getContext(), newInTypes, newResTys);
    LLVM_DEBUG(llvm::dbgs() << "new func: " << newFuncTy << '\n');
    func.setType(newFuncTy);

    for (std::pair<unsigned, mlir::NamedAttribute> extraAttr : extraAttrs)
      func.setArgAttr(extraAttr.first, extraAttr.second.getName(),
                      extraAttr.second.getValue());

    for (auto [resId, resAttrList] : resultAttrs)
      for (mlir::NamedAttribute resAttr : resAttrList)
        func.setResultAttr(resId, resAttr.getName(), resAttr.getValue());

    // Replace attributes to the correct argument if there was an argument shift
    // to the right.
    if (argumentShift > 0) {
      for (std::pair<unsigned, mlir::NamedAttribute> savedAttr : savedAttrs) {
        func.removeArgAttr(savedAttr.first, savedAttr.second.getName());
        func.setArgAttr(savedAttr.first + argumentShift,
                        savedAttr.second.getName(),
                        savedAttr.second.getValue());
      }
    }

    for (auto &fixup : fixups)
      if (fixup.finalizer)
        (*fixup.finalizer)(func);
  }

  inline bool functionArgIsSRet(unsigned index, mlir::func::FuncOp func) {
    if (auto attr = func.getArgAttrOfType<mlir::TypeAttr>(index, "llvm.sret"))
      return true;
    return false;
  }

  /// Convert a complex return value. This can involve converting the return
  /// value to a "hidden" first argument or packing the complex into a wide
  /// GPR.
  template <typename A, typename B, typename C>
  void doComplexReturn(mlir::func::FuncOp func, A cmplx, B &newResTys,
                       fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
                       C &fixups) {
    if (noComplexConversion) {
      newResTys.push_back(cmplx);
      return;
    }
    auto m =
        specifics->complexReturnType(func.getLoc(), cmplx.getElementType());
    assert(m.size() == 1);
    auto &tup = m[0];
    auto attr = std::get<fir::CodeGenSpecifics::Attributes>(tup);
    auto argTy = std::get<mlir::Type>(tup);
    if (attr.isSRet()) {
      unsigned argNo = newInTyAndAttrs.size();
      if (auto align = attr.getAlignment())
        fixups.emplace_back(
            FixupTy::Codes::ReturnAsStore, argNo, [=](mlir::func::FuncOp func) {
              auto elemType = fir::dyn_cast_ptrOrBoxEleTy(
                  func.getFunctionType().getInput(argNo));
              func.setArgAttr(argNo, "llvm.sret",
                              mlir::TypeAttr::get(elemType));
              func.setArgAttr(argNo, "llvm.align",
                              rewriter->getIntegerAttr(
                                  rewriter->getIntegerType(32), align));
            });
      else
        fixups.emplace_back(FixupTy::Codes::ReturnAsStore, argNo,
                            [=](mlir::func::FuncOp func) {
                              auto elemType = fir::dyn_cast_ptrOrBoxEleTy(
                                  func.getFunctionType().getInput(argNo));
                              func.setArgAttr(argNo, "llvm.sret",
                                              mlir::TypeAttr::get(elemType));
                            });
      newInTyAndAttrs.push_back(tup);
      return;
    }
    if (auto align = attr.getAlignment())
      fixups.emplace_back(
          FixupTy::Codes::ReturnType, newResTys.size(),
          [=](mlir::func::FuncOp func) {
            func.setArgAttr(
                newResTys.size(), "llvm.align",
                rewriter->getIntegerAttr(rewriter->getIntegerType(32), align));
          });
    else
      fixups.emplace_back(FixupTy::Codes::ReturnType, newResTys.size());
    newResTys.push_back(argTy);
  }

  template <typename FIXUPS>
  void
  createFuncOpArgFixups(mlir::func::FuncOp func,
                        fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
                        fir::CodeGenSpecifics::Marshalling &argsInTys,
                        FIXUPS &fixups) {
    const auto fixupCode = argsInTys.size() > 1 ? FixupTy::Codes::Split
                                                : FixupTy::Codes::ArgumentType;
    for (auto e : llvm::enumerate(argsInTys)) {
      auto &tup = e.value();
      auto index = e.index();
      auto attr = std::get<fir::CodeGenSpecifics::Attributes>(tup);
      auto argNo = newInTyAndAttrs.size();
      if (attr.isByVal()) {
        if (auto align = attr.getAlignment())
          fixups.emplace_back(FixupTy::Codes::ArgumentAsLoad, argNo,
                              [=](mlir::func::FuncOp func) {
                                auto elemType = fir::dyn_cast_ptrOrBoxEleTy(
                                    func.getFunctionType().getInput(argNo));
                                func.setArgAttr(argNo, "llvm.byval",
                                                mlir::TypeAttr::get(elemType));
                                func.setArgAttr(
                                    argNo, "llvm.align",
                                    rewriter->getIntegerAttr(
                                        rewriter->getIntegerType(32), align));
                              });
        else
          fixups.emplace_back(FixupTy::Codes::ArgumentAsLoad,
                              newInTyAndAttrs.size(),
                              [=](mlir::func::FuncOp func) {
                                auto elemType = fir::dyn_cast_ptrOrBoxEleTy(
                                    func.getFunctionType().getInput(argNo));
                                func.setArgAttr(argNo, "llvm.byval",
                                                mlir::TypeAttr::get(elemType));
                              });
      } else {
        if (auto align = attr.getAlignment())
          fixups.emplace_back(
              fixupCode, argNo, index, [=](mlir::func::FuncOp func) {
                func.setArgAttr(argNo, "llvm.align",
                                rewriter->getIntegerAttr(
                                    rewriter->getIntegerType(32), align));
              });
        else
          fixups.emplace_back(fixupCode, argNo, index);
      }
      newInTyAndAttrs.push_back(tup);
    }
  }

  /// Convert a complex argument value. This can involve storing the value to
  /// a temporary memory location or factoring the value into two distinct
  /// arguments.
  template <typename A, typename B>
  void doComplexArg(mlir::func::FuncOp func, A cmplx,
                    fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
                    B &fixups) {
    if (noComplexConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(cmplx));
      return;
    }
    auto cplxArgs =
        specifics->complexArgumentType(func.getLoc(), cmplx.getElementType());
    createFuncOpArgFixups(func, newInTyAndAttrs, cplxArgs, fixups);
  }

  template <typename FIXUPS>
  void doStructArg(mlir::func::FuncOp func, fir::RecordType recTy,
                   fir::CodeGenSpecifics::Marshalling &newInTyAndAttrs,
                   FIXUPS &fixups) {
    if (noStructConversion) {
      newInTyAndAttrs.push_back(fir::CodeGenSpecifics::getTypeAndAttr(recTy));
      return;
    }
    auto structArgs =
        specifics->structArgumentType(func.getLoc(), recTy, newInTyAndAttrs);
    createFuncOpArgFixups(func, newInTyAndAttrs, structArgs, fixups);
  }

private:
  // Replace `op` and remove it.
  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues) {
    op->replaceAllUsesWith(newValues);
    op->dropAllReferences();
    op->erase();
  }

  inline void setMembers(fir::CodeGenSpecifics *s, mlir::OpBuilder *r,
                         mlir::DataLayout *dl) {
    specifics = s;
    rewriter = r;
    dataLayout = dl;
  }

  inline void clearMembers() { setMembers(nullptr, nullptr, nullptr); }

  // Inserts a call to llvm.stacksave at the current insertion
  // point and the given location. Returns the call's result Value.
  inline mlir::Value genStackSave(mlir::Location loc) {
    return rewriter->create<fir::CallOp>(loc, stackSaveFn).getResult(0);
  }

  // Inserts a call to llvm.stackrestore at the current insertion
  // point and the given location and argument.
  inline void genStackRestore(mlir::Location loc, mlir::Value sp) {
    rewriter->create<fir::CallOp>(loc, stackRestoreFn, mlir::ValueRange{sp});
  }

  fir::CodeGenSpecifics *specifics = nullptr;
  mlir::OpBuilder *rewriter = nullptr;
  mlir::DataLayout *dataLayout = nullptr;
  mlir::func::FuncOp stackSaveFn = nullptr;
  mlir::func::FuncOp stackRestoreFn = nullptr;
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
fir::createFirTargetRewritePass(const fir::TargetRewriteOptions &options) {
  return std::make_unique<TargetRewrite>(options);
}
