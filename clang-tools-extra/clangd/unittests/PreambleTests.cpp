//===--- PreambleTests.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Compiler.h"
#include "Preamble.h"
#include "TestFS.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

using testing::_;
using testing::Contains;
using testing::Pair;

MATCHER_P(HasContents, Contents, "") { return arg->getBuffer() == Contents; }

TEST(PreamblePatchTest, IncludeParsing) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics Diags;
  ParseInputs PI;
  PI.FS = FS.getFileSystem();

  // We expect any line with a point to show up in the patch.
  llvm::StringRef Cases[] = {
      // Only preamble
      R"cpp(^#include "a.h")cpp",
      // Both preamble and mainfile
      R"cpp(
        ^#include "a.h"
        garbage, finishes preamble
        #include "a.h")cpp",
      // Mixed directives
      R"cpp(
        ^#include "a.h"
        #pragma directive
        // some comments
        ^#include_next <a.h>
        #ifdef skipped
        ^#import "a.h"
        #endif)cpp",
      // Broken directives
      R"cpp(
        #include "a
        ^#include "a.h"
        #include <b
        ^#include <b.h>)cpp",
  };

  // ms-compatibility changes meaning of #import, make sure it is turned off.
  CDB.ExtraClangFlags.push_back("-fno-ms-compatibility");
  const auto FileName = testPath("foo.cc");
  for (const auto Case : Cases) {
    Annotations Test(Case);
    const auto Code = Test.code();
    PI.CompileCommand = *CDB.getCompileCommand(FileName);

    SCOPED_TRACE(Code);
    // Check preamble lexing logic by building an empty preamble and patching it
    // with all the contents.
    PI.Contents = "";
    const auto CI = buildCompilerInvocation(PI, Diags);
    const auto EmptyPreamble = buildPreamble(FileName, *CI, PI, true, nullptr);
    PI.Contents = Code.str();

    std::string ExpectedBuffer;
    const auto Points = Test.points();
    for (const auto &P : Points) {
      // Copy the whole line.
      auto StartOffset = llvm::cantFail(positionToOffset(Code, P));
      ExpectedBuffer.append(Code.substr(StartOffset)
                                .take_until([](char C) { return C == '\n'; })
                                .str());
      ExpectedBuffer += '\n';
    }

    PreamblePatch::create(FileName, PI, *EmptyPreamble).apply(*CI);
    EXPECT_THAT(CI->getPreprocessorOpts().RemappedFileBuffers,
                Contains(Pair(_, HasContents(ExpectedBuffer))));
    for (const auto &RB : CI->getPreprocessorOpts().RemappedFileBuffers)
      delete RB.second;
  }
}

TEST(PreamblePatchTest, ContainsNewIncludes) {
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics Diags;
  // ms-compatibility changes meaning of #import, make sure it is turned off.
  CDB.ExtraClangFlags.push_back("-fno-ms-compatibility");

  const auto FileName = testPath("foo.cc");
  ParseInputs PI;
  PI.FS = FS.getFileSystem();
  PI.CompileCommand = *CDB.getCompileCommand(FileName);
  PI.Contents = "#include <existing.h>\n";

  // Check diffing logic by adding a new header to the preamble and ensuring
  // only it is patched.
  const auto CI = buildCompilerInvocation(PI, Diags);
  const auto FullPreamble = buildPreamble(FileName, *CI, PI, true, nullptr);

  constexpr llvm::StringLiteral Patch =
      "#include <test>\n#import <existing.h>\n";
  // We provide the same includes twice, they shouldn't be included in the
  // patch.
  PI.Contents = (Patch + PI.Contents + PI.Contents).str();
  PreamblePatch::create(FileName, PI, *FullPreamble).apply(*CI);
  EXPECT_THAT(CI->getPreprocessorOpts().RemappedFileBuffers,
              Contains(Pair(_, HasContents(Patch))));
  for (const auto &RB : CI->getPreprocessorOpts().RemappedFileBuffers)
    delete RB.second;
}

} // namespace
} // namespace clangd
} // namespace clang
