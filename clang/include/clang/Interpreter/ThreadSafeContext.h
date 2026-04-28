//===--- ThreadSafeContext.h - Thread Safe Context ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thread-safe wrapper for LLVMContext (ORC-independent).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_THREADSAFECONTEXT_H
#define LLVM_CLANG_INTERPRETER_THREADSAFECONTEXT_H

#include "llvm/IR/LLVMContext.h"
#include <memory>
#include <mutex>

namespace clang {

/// Thread-safe wrapper for LLVMContext.
/// Provides safe access to an LLVMContext across multiple threads.
class ThreadSafeContext {
private:
  struct State {
    State(std::unique_ptr<llvm::LLVMContext> Ctx) 
        : Ctx(std::move(Ctx)) {}
    
    std::unique_ptr<llvm::LLVMContext> Ctx;
    std::recursive_mutex Mutex;
  };

public:
  /// Construct a null context.
  ThreadSafeContext() = default;
  
  /// Construct a ThreadSafeContext from the given LLVMContext.
  ThreadSafeContext(std::unique_ptr<llvm::LLVMContext> NewCtx)
      : S(std::make_shared<State>(std::move(NewCtx))) {
    assert(S->Ctx != nullptr &&
           "Can not construct a ThreadSafeContext from a nullptr");
  }

  /// Locks the context and calls the given function with a pointer to it.
  template <typename Func> 
  decltype(auto) withContextDo(Func &&F) {
    if (auto TmpS = S) {
      std::lock_guard<std::recursive_mutex> Lock(TmpS->Mutex);
      return F(TmpS->Ctx.get());
    } else
      return F((llvm::LLVMContext *)nullptr);
  }

  /// Locks the context and calls the given function with a const pointer to it.
  template <typename Func> 
  decltype(auto) withContextDo(Func &&F) const {
    if (auto TmpS = S) {
      std::lock_guard<std::recursive_mutex> Lock(TmpS->Mutex);
      return F(const_cast<const llvm::LLVMContext *>(TmpS->Ctx.get()));
    } else
      return F((const llvm::LLVMContext *)nullptr);
  }

private:
  std::shared_ptr<State> S;
};

} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_THREADSAFECONTEXT_H
