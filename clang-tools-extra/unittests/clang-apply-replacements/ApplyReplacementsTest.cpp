//===- clang-apply-replacements/ApplyReplacementsTest.cpp
//----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
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
  TUs.push_back({MainSourceFile,
                 {{DiagnosticName,
                   Message,
                   Replacements,
                   {},
                   Diagnostic::Warning,
                   BuildDirectory}}});
  return TUs;
}

// Test to ensure diagnostics with no fixes, will be merged correctly
// before applying.
TEST(ApplyReplacementsTest, mergeDiagnosticsWithNoFixes) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), DiagOpts.get());
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);
  TUDiagnostics TUs =
      makeTUDiagnostics("path/to/source.cpp", "diagnostic", {}, {}, "path/to");
  FileToReplacementsMap ReplacementsMap;

  EXPECT_TRUE(mergeAndDeduplicate(TUs, ReplacementsMap, SM));
  EXPECT_TRUE(ReplacementsMap.empty());
}

} // end namespace tooling
} // end namespace clang
