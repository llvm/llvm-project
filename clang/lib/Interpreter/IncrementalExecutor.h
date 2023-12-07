//===--- IncrementalExecutor.h - Incremental Execution ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
#define LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

#include <memory>

namespace llvm {
class Error;
namespace orc {
class LLJIT;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {

struct PartialTranslationUnit;
class TargetInfo;

class IncrementalExecutor {
  using CtorDtorIterator = llvm::orc::CtorDtorIterator;
  std::unique_ptr<llvm::orc::LLJIT> Jit;
  llvm::orc::ThreadSafeContext &TSCtx;

  llvm::DenseMap<const PartialTranslationUnit *, llvm::orc::ResourceTrackerSP>
      ResourceTrackers;

public:
  enum SymbolNameKind { IRName, LinkerName };

  IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC, llvm::Error &Err,
                      const clang::TargetInfo &TI);
  ~IncrementalExecutor();

  llvm::Error addModule(PartialTranslationUnit &PTU);
  llvm::Error removeModule(PartialTranslationUnit &PTU);
  llvm::Error runCtors() const;
  llvm::Error cleanUp();
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef Name, SymbolNameKind NameKind) const;

  llvm::orc::LLJIT &GetExecutionEngine() { return *Jit; }
};

} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALEXECUTOR_H
