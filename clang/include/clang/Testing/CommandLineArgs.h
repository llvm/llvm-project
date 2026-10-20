//===--- CommandLineArgs.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines language options for Clang unittests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TESTING_COMMANDLINEARGS_H
#define LLVM_CLANG_TESTING_COMMANDLINEARGS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {

enum TestLanguage {
#define TESTLANGUAGE(lang, version, std_flag, version_index)                   \
  Lang_##lang##version,
#include "clang/Testing/TestLanguage.def"

  Lang_OpenCL,
  Lang_OBJC,
  Lang_OBJCXX,
};

std::vector<TestLanguage> getCOrLater(int MinimumStd);
std::vector<TestLanguage> getCXXOrLater(int MinimumStd);

std::vector<std::string> getCommandLineArgsForTesting(TestLanguage Lang);
std::vector<std::string> getCC1ArgsForTesting(TestLanguage Lang);

StringRef getFilenameForTesting(TestLanguage Lang);

/// Find a target name such that looking for it in TargetRegistry by that name
/// returns the same target. We expect that there is at least one target
/// configured with this property.
std::string getAnyTargetForTesting();

} // end namespace clang

#endif
