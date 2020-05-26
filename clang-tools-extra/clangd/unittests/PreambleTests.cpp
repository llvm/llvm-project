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
#include "TestTU.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <clang/Frontend/FrontendActions.h>
#include <string>
#include <vector>

using testing::Field;

namespace clang {
namespace clangd {
namespace {

// Builds a preamble for BaselineContents, patches it for ModifiedContents and
// returns the includes in the patch.
IncludeStructure
collectPatchedIncludes(llvm::StringRef ModifiedContents,
                       llvm::StringRef BaselineContents,
                       llvm::StringRef MainFileName = "main.cpp") {
  std::string MainFile = testPath(MainFileName);
  ParseInputs PI;
  PI.FS = new llvm::vfs::InMemoryFileSystem;
  MockCompilationDatabase CDB;
  // ms-compatibility changes meaning of #import, make sure it is turned off.
  CDB.ExtraClangFlags.push_back("-fno-ms-compatibility");
  PI.CompileCommand = CDB.getCompileCommand(MainFile).getValue();
  // Create invocation
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(PI, Diags);
  assert(CI && "failed to create compiler invocation");
  // Build baseline preamble.
  PI.Contents = BaselineContents.str();
  PI.Version = "baseline preamble";
  auto BaselinePreamble = buildPreamble(MainFile, *CI, PI, true, nullptr);
  assert(BaselinePreamble && "failed to build baseline preamble");
  // Create the patch.
  PI.Contents = ModifiedContents.str();
  PI.Version = "modified contents";
  auto PP = PreamblePatch::create(MainFile, PI, *BaselinePreamble);
  // Collect patch contents.
  PP.apply(*CI);
  llvm::StringRef PatchContents;
  for (const auto &Rempaped : CI->getPreprocessorOpts().RemappedFileBuffers) {
    if (Rempaped.first == testPath("__preamble_patch__.h")) {
      PatchContents = Rempaped.second->getBuffer();
      break;
    }
  }
  // Run preprocessor over the modified contents with patched Invocation to and
  // BaselinePreamble to collect includes in the patch. We trim the input to
  // only preamble section to not collect includes in the mainfile.
  auto Bounds = Lexer::ComputePreamble(ModifiedContents, *CI->getLangOpts());
  auto Clang =
      prepareCompilerInstance(std::move(CI), &BaselinePreamble->Preamble,
                              llvm::MemoryBuffer::getMemBufferCopy(
                                  ModifiedContents.slice(0, Bounds.Size).str()),
                              PI.FS, Diags);
  Clang->getPreprocessorOpts().ImplicitPCHInclude.clear();
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    ADD_FAILURE() << "failed begin source file";
    return {};
  }
  IncludeStructure Includes;
  Clang->getPreprocessor().addPPCallbacks(
      collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));
  if (llvm::Error Err = Action.Execute()) {
    ADD_FAILURE() << "failed to execute action: " << std::move(Err);
    return {};
  }
  Action.EndSourceFile();
  return Includes;
}

// Check preamble lexing logic by building an empty preamble and patching it
// with all the contents.
TEST(PreamblePatchTest, IncludeParsing) {
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
      // Directive is not part of preamble if it is not the token immediately
      // followed by the hash (#).
      R"cpp(
        ^#include "a.h"
        #/**/include <b.h>)cpp",
  };

  for (const auto Case : Cases) {
    Annotations Test(Case);
    const auto Code = Test.code();
    SCOPED_TRACE(Code);

    auto Includes =
        collectPatchedIncludes(Code, /*BaselineContents=*/"").MainFileIncludes;
    auto Points = Test.points();
    ASSERT_EQ(Includes.size(), Points.size());
    for (size_t I = 0, E = Includes.size(); I != E; ++I)
      EXPECT_EQ(Includes[I].HashLine, Points[I].line);
  }
}

TEST(PreamblePatchTest, ContainsNewIncludes) {
  constexpr llvm::StringLiteral BaselineContents = R"cpp(
    #include <a.h>
    #include <b.h> // This will be removed
    #include <c.h>
  )cpp";
  constexpr llvm::StringLiteral ModifiedContents = R"cpp(
    #include <a.h>
    #include <c.h> // This has changed a line.
    #include <c.h> // This is a duplicate.
    #include <d.h> // This is newly introduced.
  )cpp";
  auto Includes = collectPatchedIncludes(ModifiedContents, BaselineContents)
                      .MainFileIncludes;
  EXPECT_THAT(Includes, ElementsAre(AllOf(Field(&Inclusion::Written, "<d.h>"),
                                          Field(&Inclusion::HashLine, 4))));
}

TEST(PreamblePatchTest, MainFileIsEscaped) {
  auto Includes = collectPatchedIncludes("#include <a.h>", "", "file\"name.cpp")
                      .MainFileIncludes;
  EXPECT_THAT(Includes, ElementsAre(AllOf(Field(&Inclusion::Written, "<a.h>"),
                                          Field(&Inclusion::HashLine, 0))));
}
} // namespace
} // namespace clangd
} // namespace clang
