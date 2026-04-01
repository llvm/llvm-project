//===- Tester.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tester class used in the AIIR Reduce tool.
//
// A Tester object is passed as an argument to the reduction passes and it is
// used to run the interestingness testing script on the different generated
// reduced variants of the test case.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_REDUCER_TESTER_H
#define AIIR_REDUCER_TESTER_H

#include "aiir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace aiir {

/// This class is used to keep track of the testing environment of the tool. It
/// contains a method to run the interestingness testing script on a AIIR test
/// case file.
class Tester {
public:
  enum class Interestingness {
    True,
    False,
    Untested,
  };

  Tester() = default;
  Tester(const Tester &) = default;

  Tester(StringRef testScript, ArrayRef<std::string> testScriptArgs);

  /// Runs the interestingness testing script on a AIIR test case file. Returns
  /// true if the interesting behavior is present in the test case or false
  /// otherwise.
  std::pair<Interestingness, size_t> isInteresting(ModuleOp module) const;

  /// Return whether the file in the given path is interesting.
  Interestingness isInteresting(StringRef testCase) const;

  void setTestScript(StringRef script) { testScript = script; }
  void setTestScriptArgs(ArrayRef<std::string> args) { testScriptArgs = args; }

private:
  StringRef testScript;
  ArrayRef<std::string> testScriptArgs;
};

} // namespace aiir

#endif
