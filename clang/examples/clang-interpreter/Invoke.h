//===-- examples/clang-interpreter/Invoke.h - Clang C Interpreter Example -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_EXAMPLE_INTERPRETER_INVOKE_H
#define CLANG_EXAMPLE_INTERPRETER_INVOKE_H

namespace llvm {
  class ExecutionEngine;
  class Function;
}

#include <string>
#include <vector>

namespace interpreter {

typedef std::vector<std::string> InvokeArgs;

typedef int (*Invoker)(llvm::ExecutionEngine *EE, llvm::Function *EntryFn,
                       const InvokeArgs &Args, char *const *EnvP);

int TryIt(llvm::ExecutionEngine *EE, llvm::Function *EntryFn,
          const InvokeArgs &Args, char *const *EnvP,
          Invoker Invoke);

} // interpreter

#endif // CLANG_EXAMPLE_INTERPRETER_INVOKE_H
