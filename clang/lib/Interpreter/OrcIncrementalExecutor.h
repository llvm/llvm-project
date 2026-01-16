//===--- OrcIncrementalExecutor.h - Orc Incremental Execution ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an Orc-based incremental code execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_ORCINCREMENTALEXECUTOR_H
#define LLVM_CLANG_LIB_INTERPRETER_ORCINCREMENTALEXECUTOR_H

#include "clang/Interpreter/IncrementalExecutor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace llvm {
class Error;
namespace orc {
class JITTargetMachineBuilder;
class LLJIT;
class LLJITBuilder;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {

struct PartialTranslationUnit;

class OrcIncrementalExecutor : public IncrementalExecutor {
  std::unique_ptr<llvm::orc::LLJIT> Jit;
  llvm::orc::ThreadSafeContext &TSCtx;

  llvm::DenseMap<const PartialTranslationUnit *, llvm::orc::ResourceTrackerSP>
      ResourceTrackers;

protected:
  OrcIncrementalExecutor(llvm::orc::ThreadSafeContext &TSC);

public:
  OrcIncrementalExecutor(llvm::orc::ThreadSafeContext &TSC,
                         llvm::orc::LLJITBuilder &JITBuilder, llvm::Error &Err);
  ~OrcIncrementalExecutor() override;

  llvm::Error addModule(PartialTranslationUnit &PTU) override;
  llvm::Error removeModule(PartialTranslationUnit &PTU) override;
  llvm::Error runCtors() const override;
  llvm::Error cleanUp() override;
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef Name,
                   SymbolNameKind NameKind) const override;
  llvm::Error LoadDynamicLibrary(const char *name) override;
};

} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_ORCINCREMENTALEXECUTOR_H
