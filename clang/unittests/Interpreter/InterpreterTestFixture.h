//===- unittests/Interpreter/InterpreterTestBase.h ------------------ C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_INTERPRETER_INTERPRETERTESTBASE_H
#define LLVM_CLANG_UNITTESTS_INTERPRETER_INTERPRETERTESTBASE_H

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#if defined(_AIX) || defined(__MVS__)
#define CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
#endif

namespace clang {

class InterpreterTestBase : public ::testing::Test {
protected:
  static bool HostSupportsJIT() {
#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
    return false;
#else
    if (auto JIT = llvm::orc::LLJITBuilder().create()) {
      return true;
    } else {
      llvm::consumeError(JIT.takeError());
      return false;
    }
#endif
  }

  void SetUp() override {
// FIXME : WebAssembly doesn't currently support Jit (see
// https: // github.com/llvm/llvm-project/pull/150977#discussion_r2237521095).
// so this check of HostSupportsJIT has been skipped
// over until support is added, and HostSupportsJIT can return true.
#ifndef __EMSCRIPTEN__
    if (!HostSupportsJIT())
      GTEST_SKIP();
#endif
  }

  void TearDown() override {}

  static void SetUpTestSuite() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }

  static void TearDownTestSuite() { llvm::llvm_shutdown(); }
};

} // namespace clang

#undef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT

#endif // LLVM_CLANG_UNITTESTS_INTERPRETER_INTERPRETERTESTBASE_H
