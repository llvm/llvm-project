//===-- CUFAddConstructor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/CUDA/registration.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFADDCONSTRUCTOR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace Fortran::runtime::cuda;

namespace {

static constexpr llvm::StringRef cudaFortranCtorName{
    "__cudaFortranConstructor"};
static constexpr llvm::StringRef managedPtrSuffix{".managed.ptr"};

/// Create an 8-byte pointer global in the __nv_managed_data__ section.
/// The CUDA runtime populates this pointer with the unified memory address
/// when the module is initialized via __cudaInitModule.
static fir::GlobalOp createManagedPointerGlobal(fir::FirOpBuilder &builder,
                                                mlir::ModuleOp mod,
                                                fir::GlobalOp globalOp) {
  mlir::MLIRContext *ctx = mod.getContext();
  std::string ptrGlobalName = (globalOp.getSymName() + managedPtrSuffix).str();
  auto ptrTy = fir::LLVMPointerType::get(ctx, mlir::IntegerType::get(ctx, 8));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(globalOp);

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, "section"),
                           mlir::StringAttr::get(ctx, "__nv_managed_data__")));

  mlir::DenseElementsAttr initAttr = {};
  auto ptrGlobal = fir::GlobalOp::create(
      builder, globalOp.getLoc(), ptrGlobalName, /*isConstant=*/false,
      /*isTarget=*/false, ptrTy, initAttr,
      /*linkName=*/builder.createInternalLinkage(), attrs);

  mlir::Region &region = ptrGlobal.getRegion();
  mlir::Block *block = builder.createBlock(&region);
  builder.setInsertionPointToStart(block);
  mlir::Value zero = fir::ZeroOp::create(builder, globalOp.getLoc(), ptrTy);
  fir::HasValueOp::create(builder, globalOp.getLoc(), zero);

  return ptrGlobal;
}

/// Return true if \p hostGlobal is a host module-scope global that has been
/// mirrored in the GPU module as an external (no-body) declaration by the
/// CUFDeviceGlobal pass under -gpu=mem:unified. Such globals must be
/// registered with the CUDA driver via CUFRegisterExternalVariable so the
/// device-side `.extern` symbol resolves to the host pointer at module-load
/// time and HMM/ATS handles migration.
static bool isCudaUnifiedExternalGlobal(fir::GlobalOp hostGlobal,
                                        mlir::SymbolTable &gpuSymTable) {
  if (hostGlobal.getDataAttrAttr())
    return false;
  if (hostGlobal.getConstant())
    return false;
  auto gpuGlobal = gpuSymTable.lookup<fir::GlobalOp>(hostGlobal.getSymName());
  if (!gpuGlobal)
    return false;
  return !gpuGlobal.isInitialized();
}

/// Build a C-style name literal (`<symname>\0`) for use as the deviceName
/// argument of a CUF registration runtime call.
static mlir::Value buildGlobalNameLiteral(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          fir::GlobalOp globalOp) {
  std::string nameStr = globalOp.getSymbol().getValue().str();
  nameStr += '\0';
  return fir::getBase(fir::factory::createStringLiteral(builder, loc, nameStr));
}

/// Compute the storage size in bytes of \p globalOp. For a box-typed
/// allocatable global the size is the descriptor size (after type
/// conversion); otherwise it's the size of the global's declared type.
static mlir::Value computeGlobalSize(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type idxTy,
                                     const mlir::DataLayout &dl,
                                     const fir::KindMapping &kindMap,
                                     fir::LLVMTypeConverter &typeConverter,
                                     fir::GlobalOp globalOp) {
  std::optional<uint64_t> size;
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(globalOp.getType())) {
    mlir::Type structTy = typeConverter.convertBoxTypeAsStruct(boxTy);
    size = dl.getTypeSizeInBits(structTy) / 8;
  }
  if (!size) {
    size = fir::getTypeSizeAndAlignmentOrCrash(loc, globalOp.getType(), dl,
                                               kindMap)
               .first;
  }
  return builder.createIntegerConstant(loc, idxTy, *size);
}

/// Emit a call to a CUF registration runtime function with the canonical
/// (module, addr, name, size) signature, where addr is the address of \p
/// addrGlobal taken via fir.address_of and name/size describe \p nameGlobal.
/// Used both for CUFRegisterVariable / CUFRegisterManagedVariable / and
/// CUFRegisterExternalVariable.
static void
emitCUFRegistrationCall(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Type idxTy, const mlir::DataLayout &dl,
                        const fir::KindMapping &kindMap,
                        fir::LLVMTypeConverter &typeConverter,
                        mlir::Value registeredMod, mlir::func::FuncOp func,
                        fir::GlobalOp addrGlobal, fir::GlobalOp nameGlobal) {
  mlir::Value gblName = buildGlobalNameLiteral(builder, loc, nameGlobal);
  mlir::Value sizeVal = computeGlobalSize(builder, loc, idxTy, dl, kindMap,
                                          typeConverter, nameGlobal);
  mlir::Value addr = fir::AddrOfOp::create(
      builder, loc, addrGlobal.resultType(), addrGlobal.getSymbol());
  llvm::SmallVector<mlir::Value> args{
      fir::runtime::createArguments(builder, loc, func.getFunctionType(),
                                    registeredMod, addr, gblName, sizeVal)};
  fir::CallOp::create(builder, loc, func, args);
}

static bool hasRegisteredGlobals(mlir::ModuleOp mod,
                                 mlir::SymbolTable gpuSymTable,
                                 bool cudaUnified) {
  for (fir::GlobalOp globalOp : mod.getOps<fir::GlobalOp>()) {
    auto attr = globalOp.getDataAttrAttr();
    if (!attr) {
      if (cudaUnified && isCudaUnifiedExternalGlobal(globalOp, gpuSymTable))
        return true;
      continue;
    }
    if (!gpuSymTable.lookup(globalOp.getSymName()))
      continue;
    if (attr.getValue() == cuf::DataAttribute::Managed &&
        !mlir::isa<fir::BaseBoxType>(globalOp.getType()))
      return true;
    switch (attr.getValue()) {
    case cuf::DataAttribute::Device:
    case cuf::DataAttribute::Constant:
    case cuf::DataAttribute::Managed: {
      return true;
    } break;
    default:
      break;
    }
  }
  return false;
}

static bool hasKernel(mlir::gpu::GPUModuleOp gpuMod) {
  for (auto func : gpuMod.getOps<mlir::gpu::GPUFuncOp>())
    if (func.isKernel())
      return true;
  return false;
}

struct CUFAddConstructor
    : public fir::impl::CUFAddConstructorBase<CUFAddConstructor> {

  using CUFAddConstructorBase::CUFAddConstructorBase;

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::SymbolTable symTab(mod);
    mlir::OpBuilder opBuilder{mod.getBodyRegion()};
    fir::FirOpBuilder builder(opBuilder, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    builder.setInsertionPointToEnd(mod.getBody());
    mlir::Location loc = mod.getLoc();
    auto *ctx = mod.getContext();
    auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto idxTy = builder.getIndexType();
    auto funcTy =
        mlir::LLVM::LLVMFunctionType::get(voidTy, {}, /*isVarArg=*/false);
    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetMLIRDataLayout(mod, /*allowDefaultLayout=*/false);
    if (!dl) {
      mlir::emitError(mod.getLoc(),
                      "data layout attribute is required to perform " +
                          getName() + "pass");
    }

    // Symbol reference to CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    auto registerFuncOp = mlir::LLVM::LLVMFuncOp::create(
        builder, loc, RTNAME_STRING(CUFRegisterAllocator), funcTy);
    registerFuncOp.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto cufRegisterAllocatorRef = mlir::SymbolRefAttr::get(
        mod.getContext(), RTNAME_STRING(CUFRegisterAllocator));
    builder.setInsertionPointToEnd(mod.getBody());

    // Create the constructor function that call CUFRegisterAllocator.
    auto func = mlir::LLVM::LLVMFuncOp::create(builder, loc,
                                               cudaFortranCtorName, funcTy);
    func.setLinkage(mlir::LLVM::Linkage::Internal);
    builder.setInsertionPointToStart(func.addEntryBlock(builder));
    mlir::LLVM::CallOp::create(builder, loc, funcTy, cufRegisterAllocatorRef);

    auto gpuMod = symTab.lookup<mlir::gpu::GPUModuleOp>(cudaDeviceModuleName);
    if (gpuMod) {
      mlir::SymbolTable gpuSymTable(gpuMod);
      bool needsModuleRegistration =
          hasKernel(gpuMod) ||
          hasRegisteredGlobals(mod, gpuSymTable, cudaUnified);
      if (needsModuleRegistration) {
        auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
        auto registeredMod = cuf::RegisterModuleOp::create(
            builder, loc, llvmPtrTy,
            mlir::SymbolRefAttr::get(ctx, gpuMod.getName()));

        fir::LLVMTypeConverter typeConverter(
            mod, /*applyTBAA=*/false, /*forceUnifiedTBAATree=*/false, *dl);
        // Register kernels
        for (auto func : gpuMod.getOps<mlir::gpu::GPUFuncOp>()) {
          if (func.isKernel()) {
            auto kernelName = mlir::SymbolRefAttr::get(
                builder.getStringAttr(cudaDeviceModuleName),
                {mlir::SymbolRefAttr::get(builder.getContext(),
                                          func.getName())});
            cuf::RegisterKernelOp::create(builder, loc, kernelName,
                                          registeredMod);
          }
        }

        // Register variables
        bool hasNonAllocManagedGlobal = false;
        for (fir::GlobalOp globalOp : mod.getOps<fir::GlobalOp>()) {
          auto attr = globalOp.getDataAttrAttr();
          if (!attr)
            continue;
          if (!gpuSymTable.lookup(globalOp.getSymName()))
            continue;

          bool isNonAllocManagedGlobal =
              attr.getValue() == cuf::DataAttribute::Managed &&
              !mlir::isa<fir::BaseBoxType>(globalOp.getType());

          switch (attr.getValue()) {
          case cuf::DataAttribute::Device:
          case cuf::DataAttribute::Constant:
          case cuf::DataAttribute::Managed: {
            if (isNonAllocManagedGlobal) {
              hasNonAllocManagedGlobal = true;
              // Non-allocatable managed globals use pointer indirection:
              // a companion pointer in __nv_managed_data__ holds the unified
              // memory address, registered via __cudaRegisterManagedVar.
              fir::GlobalOp ptrGlobal =
                  createManagedPointerGlobal(builder, mod, globalOp);
              auto func = fir::runtime::getRuntimeFunc<mkRTKey(
                  CUFRegisterManagedVariable)>(loc, builder);
              emitCUFRegistrationCall(builder, loc, idxTy, *dl, kindMap,
                                      typeConverter, registeredMod, func,
                                      /*addrGlobal=*/ptrGlobal,
                                      /*nameGlobal=*/globalOp);
            } else {
              auto func =
                  fir::runtime::getRuntimeFunc<mkRTKey(CUFRegisterVariable)>(
                      loc, builder);
              emitCUFRegistrationCall(builder, loc, idxTy, *dl, kindMap,
                                      typeConverter, registeredMod, func,
                                      /*addrGlobal=*/globalOp,
                                      /*nameGlobal=*/globalOp);
            }
          } break;
          default:
            break;
          }
        }

        // Register externally-linked module globals under -gpu=mem:unified.
        // CUFDeviceGlobal cloned them into the GPU module with external
        // linkage so PTX emits .extern; the CUDA driver patches the device
        // reference to the host pointer at module-load time after this call.
        // Works uniformly for fixed-shape (e.g. fir.array<5xi32>) and
        // allocatable (fir.box<fir.heap<...>>) module globals.
        if (cudaUnified) {
          for (fir::GlobalOp globalOp : mod.getOps<fir::GlobalOp>()) {
            if (!isCudaUnifiedExternalGlobal(globalOp, gpuSymTable))
              continue;
            auto func = fir::runtime::getRuntimeFunc<mkRTKey(
                CUFRegisterExternalVariable)>(loc, builder);
            emitCUFRegistrationCall(builder, loc, idxTy, *dl, kindMap,
                                    typeConverter, registeredMod, func,
                                    /*addrGlobal=*/globalOp,
                                    /*nameGlobal=*/globalOp);
          }
        }

        if (hasNonAllocManagedGlobal) {
          // Initialize the module after all variables are registered so the
          // runtime populates managed variable unified memory pointers.
          mlir::func::FuncOp initFunc =
              fir::runtime::getRuntimeFunc<mkRTKey(CUFInitModule)>(loc,
                                                                   builder);
          mlir::FunctionType initFTy = initFunc.getFunctionType();
          llvm::SmallVector<mlir::Value> initArgs{fir::runtime::createArguments(
              builder, loc, initFTy, registeredMod)};
          fir::CallOp::create(builder, loc, initFunc, initArgs);
        }
      }
    }
    mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});

    // Create the llvm.global_ctor with the function.
    // TODO: We might want to have a utility that retrieve it if already
    // created and adds new functions.
    builder.setInsertionPointToEnd(mod.getBody());
    llvm::SmallVector<mlir::Attribute> funcs;
    funcs.push_back(
        mlir::FlatSymbolRefAttr::get(mod.getContext(), func.getSymName()));
    llvm::SmallVector<int> priorities;
    llvm::SmallVector<mlir::Attribute> data;
    priorities.push_back(0);
    data.push_back(mlir::LLVM::ZeroAttr::get(mod.getContext()));
    mlir::LLVM::GlobalCtorsOp::create(
        builder, mod.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities), builder.getArrayAttr(data));
  }
};

} // end anonymous namespace
