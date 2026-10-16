//===--- CommandLineArgs.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Testing/CommandLineArgs.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
std::vector<TestLanguage> getCOrLater(const int MinimumStd) {
  std::vector<TestLanguage> Result{};

#define TESTLANGUAGE_C(lang, version, std_flag, version_index)                 \
  if (version >= MinimumStd)                                                   \
    Result.push_back(Lang_##lang##version);
#include "clang/Testing/TestLanguage.def"

  return Result;
}
std::vector<TestLanguage> getCXXOrLater(const int MinimumStd) {
  std::vector<TestLanguage> Result{};

#define TESTLANGUAGE_CXX(lang, version, std_flag, version_index)               \
  if (version >= MinimumStd)                                                   \
    Result.push_back(Lang_##lang##version);
#include "clang/Testing/TestLanguage.def"

  return Result;
}

std::vector<std::string> getCommandLineArgsForTesting(TestLanguage Lang) {
  // Test with basic arguments.
  switch (Lang) {
#define TESTLANGUAGE_C(lang, version, std_flag, version_index)                 \
  case Lang_##lang##version:                                                   \
    return { "-x", "c", "-std=" #std_flag };
#define TESTLANGUAGE_CXX(lang, version, std_flag, version_index)               \
  case Lang_##lang##version:                                                   \
    return { "-std=" #std_flag, "-frtti" };
#include "clang/Testing/TestLanguage.def"

  case Lang_OBJC:
    return {"-x", "objective-c", "-frtti", "-fobjc-nonfragile-abi"};
  case Lang_OBJCXX:
    return {"-x", "objective-c++", "-frtti"};
  case Lang_OpenCL:
    llvm_unreachable("Unhandled TestLanguage enum");
  }
  llvm_unreachable("Unhandled TestLanguage enum");
}

std::vector<std::string> getCC1ArgsForTesting(TestLanguage Lang) {
  switch (Lang) {
#define TESTLANGUAGE_C(lang, version, std_flag, version_index)                 \
  case Lang_##lang##version:                                                   \
    return { "-xc", "-std=" #std_flag };
#define TESTLANGUAGE_CXX(lang, version, std_flag, version_index)               \
  case Lang_##lang##version:                                                   \
    return { "-std=" #std_flag };
#include "clang/Testing/TestLanguage.def"

  case Lang_OBJC:
    return {"-xobjective-c"};
    break;
  case Lang_OBJCXX:
    return {"-xobjective-c++"};
    break;
  case Lang_OpenCL:
    llvm_unreachable("Unhandled TestLanguage enum");
  }
  llvm_unreachable("Unhandled TestLanguage enum");
}

StringRef getFilenameForTesting(TestLanguage Lang) {
  switch (Lang) {
#define TESTLANGUAGE_C(lang, version, std_flag, version_index)                 \
  case Lang_##lang##version:                                                   \
    return "input.c";
#define TESTLANGUAGE_CXX(lang, version, std_flag, version_index)               \
  case Lang_##lang##version:                                                   \
    return "input.cc";
#include "clang/Testing/TestLanguage.def"

  case Lang_OpenCL:
    return "input.cl";

  case Lang_OBJC:
    return "input.m";

  case Lang_OBJCXX:
    return "input.mm";
  }
  llvm_unreachable("Unhandled TestLanguage enum");
}

std::string getAnyTargetForTesting() {
  for (const auto &Target : llvm::TargetRegistry::targets()) {
    std::string Error;
    StringRef TargetName(Target.getName());
    if (TargetName == "x86-64")
      TargetName = "x86_64";
    if (llvm::TargetRegistry::lookupTarget(llvm::Triple(TargetName), Error) ==
        &Target) {
      return std::string(TargetName);
    }
  }
  return "";
}

} // end namespace clang
