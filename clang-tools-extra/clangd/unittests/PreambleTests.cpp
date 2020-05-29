//===--- PreambleTests.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Compiler.h"
#include "Headers.h"
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
#include <memory>
#include <string>
#include <vector>

using testing::Field;

namespace clang {
namespace clangd {
namespace {

MATCHER_P2(Distance, File, D, "") {
  return arg.first() == File && arg.second == D;
}

// Builds a preamble for BaselineContents, patches it for ModifiedContents and
// returns the includes in the patch.
IncludeStructure
collectPatchedIncludes(llvm::StringRef ModifiedContents,
                       llvm::StringRef BaselineContents,
                       llvm::StringRef MainFileName = "main.cpp") {
  auto TU = TestTU::withCode(BaselineContents);
  TU.Filename = MainFileName.str();
  // ms-compatibility changes meaning of #import, make sure it is turned off.
  TU.ExtraArgs = {"-fno-ms-compatibility"};
  auto BaselinePreamble = TU.preamble();
  // Create the patch.
  TU = TestTU::withCode(ModifiedContents);
  auto PI = TU.inputs();
  auto PP = PreamblePatch::create(testPath(TU.Filename), PI, *BaselinePreamble);
  // Collect patch contents.
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(PI, Diags);
  PP.apply(*CI);
  // Run preprocessor over the modified contents with patched Invocation. We
  // provide a preamble and trim contents to ensure only the implicit header
  // introduced by the patch is parsed and nothing else.
  // We don't run PP directly over the patch cotents to test production
  // behaviour.
  auto Bounds = Lexer::ComputePreamble(ModifiedContents, *CI->getLangOpts());
  auto Clang =
      prepareCompilerInstance(std::move(CI), &BaselinePreamble->Preamble,
                              llvm::MemoryBuffer::getMemBufferCopy(
                                  ModifiedContents.slice(0, Bounds.Size).str()),
                              PI.FS, Diags);
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

TEST(PreamblePatchTest, PatchesPreambleIncludes) {
  IgnoreDiagnostics Diags;
  auto TU = TestTU::withCode(R"cpp(
    #include "a.h"
    #include "c.h"
  )cpp");
  TU.AdditionalFiles["a.h"] = "#include \"b.h\"";
  TU.AdditionalFiles["b.h"] = "";
  TU.AdditionalFiles["c.h"] = "";
  auto PI = TU.inputs();
  auto BaselinePreamble = buildPreamble(
      TU.Filename, *buildCompilerInvocation(PI, Diags), PI, true, nullptr);
  // We drop c.h from modified and add a new header. Since the latter is patched
  // we should only get a.h in preamble includes.
  TU.Code = R"cpp(
    #include "a.h"
    #include "b.h"
  )cpp";
  auto PP = PreamblePatch::create(testPath(TU.Filename), TU.inputs(),
                                  *BaselinePreamble);
  // Only a.h should exists in the preamble, as c.h has been dropped and b.h was
  // newly introduced.
  EXPECT_THAT(PP.preambleIncludes(),
              ElementsAre(AllOf(Field(&Inclusion::Written, "\"a.h\""),
                                Field(&Inclusion::Resolved, testPath("a.h")))));
}
} // namespace
} // namespace clangd
} // namespace clang
