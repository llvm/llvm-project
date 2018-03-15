//==-- examples/clang-interpreter/Invoke.cpp - Clang C Interpreter Example -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Invoke.h"

#include <iostream>
#include <stdexcept>

namespace interpreter {

int TryIt(llvm::ExecutionEngine *EE, llvm::Function *EntryFn,
          const std::vector<std::string> &Args, char *const *EnvP,
          Invoker Invoke) {
  int Res = -1;
  try {
    Res = Invoke(EE, EntryFn, Args, EnvP);
  } catch (const std::exception &E) {
    std::cout << "Caught '" << E.what() << "'\n";
  } catch (...) {
    std::cout << "Unknown exception\n";
  }
  return Res;
}

}
