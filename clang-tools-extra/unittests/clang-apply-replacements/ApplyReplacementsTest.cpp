//===- clang-apply-replacements/ApplyReplacementsTest.cpp
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Format/Format.h"
#include "gtest/gtest.h"

using namespace clang::replace;
using namespace llvm;

namespace clang {
namespace tooling {

static TUDiagnostics
makeTUDiagnostics(const std::string &MainSourceFile, StringRef DiagnosticName,
                  const DiagnosticMessage &Message,
                  const StringMap<Replacements> &Replacements,
                  StringRef BuildDirectory) {
  TUDiagnostics TUs;
  TUs.push_back(
      {MainSourceFile,
       {{DiagnosticName, Message, {}, Diagnostic::Warning, BuildDirectory}}});
  return TUs;
}

// Test to ensure diagnostics with no fixes, will be merged correctly
// before applying.
TEST(ApplyReplacementsTest, mergeDiagnosticsWithNoFixes) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diagnostics(DiagnosticIDs::create(), DiagOpts);
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  TUReplacements TURs;
  TUDiagnostics TUs =
      makeTUDiagnostics("path/to/source.cpp", "diagnostic", {}, {}, "path/to");
  FileToChangesMap ReplacementsMap;

  EXPECT_TRUE(mergeAndDeduplicate(TURs, TUs, ReplacementsMap, SM));
  EXPECT_TRUE(ReplacementsMap.empty());
}

} // end namespace tooling
} // end namespace clang
