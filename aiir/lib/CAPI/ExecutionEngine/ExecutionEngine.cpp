//===- ExecutionEngine.cpp - C API for AIIR JIT ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/ExecutionEngine.h"
#include "aiir/CAPI/ExecutionEngine.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/ExecutionEngine/OptUtils.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"

using namespace aiir;

extern "C" AiirExecutionEngine
aiirExecutionEngineCreate(AiirModule op, int optLevel, int numPaths,
                          const AiirStringRef *sharedLibPaths,
                          bool enableObjectDump, bool enablePIC) {
  static bool initOnce = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  auto &ctx = *unwrap(op)->getContext();
  aiir::registerBuiltinDialectTranslation(ctx);
  aiir::registerLLVMDialectTranslation(ctx);
  aiir::registerOpenMPDialectTranslation(ctx);

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    consumeError(tmBuilderOrError.takeError());
    return AiirExecutionEngine{nullptr};
  }
  if (enablePIC)
    tmBuilderOrError->setRelocationModel(llvm::Reloc::PIC_);
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    consumeError(tmOrError.takeError());
    return AiirExecutionEngine{nullptr};
  }

  SmallVector<StringRef> libPaths;
  for (unsigned i = 0; i < static_cast<unsigned>(numPaths); ++i)
    libPaths.push_back(sharedLibPaths[i].data);

  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  auto transformer = aiir::makeOptimizingTransformer(
      optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = transformer;
  jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
  jitOptions.sharedLibPaths = libPaths;
  jitOptions.enableObjectDump = enableObjectDump;
  auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions,
                                            std::move(tmOrError.get()));
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return AiirExecutionEngine{nullptr};
  }
  return wrap(jitOrError->release());
}

extern "C" void aiirExecutionEngineInitialize(AiirExecutionEngine jit) {
  unwrap(jit)->initialize();
}

extern "C" void aiirExecutionEngineDestroy(AiirExecutionEngine jit) {
  delete (unwrap(jit));
}

extern "C" AiirLogicalResult
aiirExecutionEngineInvokePacked(AiirExecutionEngine jit, AiirStringRef name,
                                void **arguments) {
  const std::string ifaceName = ("_aiir_ciface_" + unwrap(name)).str();
  llvm::Error error = unwrap(jit)->invokePacked(
      ifaceName, MutableArrayRef<void *>{arguments, (size_t)0});
  if (error)
    return wrap(failure());
  return wrap(success());
}

extern "C" void *aiirExecutionEngineLookupPacked(AiirExecutionEngine jit,
                                                 AiirStringRef name) {
  auto optionalFPtr =
      llvm::expectedToOptional(unwrap(jit)->lookupPacked(unwrap(name)));
  if (!optionalFPtr)
    return nullptr;
  return reinterpret_cast<void *>(*optionalFPtr);
}

extern "C" void *aiirExecutionEngineLookup(AiirExecutionEngine jit,
                                           AiirStringRef name) {
  auto optionalFPtr =
      llvm::expectedToOptional(unwrap(jit)->lookup(unwrap(name)));
  if (!optionalFPtr)
    return nullptr;
  return *optionalFPtr;
}

extern "C" void aiirExecutionEngineRegisterSymbol(AiirExecutionEngine jit,
                                                  AiirStringRef name,
                                                  void *sym) {
  unwrap(jit)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner(unwrap(name))] = {llvm::orc::ExecutorAddr::fromPtr(sym),
                                         llvm::JITSymbolFlags::Exported};
    return symbolMap;
  });
}

extern "C" void aiirExecutionEngineDumpToObjectFile(AiirExecutionEngine jit,
                                                    AiirStringRef name) {
  unwrap(jit)->dumpToObjectFile(unwrap(name));
}
