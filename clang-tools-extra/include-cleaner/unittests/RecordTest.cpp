//===-- RecordTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <optional>
#include <utility>

namespace clang::include_cleaner {
namespace {
using testing::ElementsAreArray;
using testing::IsEmpty;

// Matches a Decl* if it is a NamedDecl with the given name.
MATCHER_P(named, N, "") {
  if (const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(arg)) {
    if (N == ND->getNameAsString())
      return true;
  }
  std::string S;
  llvm::raw_string_ostream OS(S);
  arg->dump(OS);
  *result_listener << S;
  return false;
}

MATCHER_P(FileNamed, N, "") {
  if (arg.getFileEntry().tryGetRealPathName() == N)
    return true;
  *result_listener << arg.getFileEntry().tryGetRealPathName().str();
  return false;
}

class RecordASTTest : public ::testing::Test {
protected:
  TestInputs Inputs;
  RecordedAST Recorded;

  RecordASTTest() {
    struct RecordAction : public ASTFrontendAction {
      RecordedAST &Out;
      RecordAction(RecordedAST &Out) : Out(Out) {}
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                     StringRef) override {
        return Out.record();
      }
    };
    Inputs.MakeAction = [this] {
      return std::make_unique<RecordAction>(Recorded);
    };
  }

  TestAST build() { return TestAST(Inputs); }
};

// Top-level decl from the main file is a root, nested ones aren't.
TEST_F(RecordASTTest, Namespace) {
  Inputs.Code =
      R"cpp(
      namespace ns {
        int x;
        namespace {
          int y;
        }
      }
    )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(named("ns")));
}

// Decl in included file is not a root.
TEST_F(RecordASTTest, Inclusion) {
  Inputs.ExtraFiles["header.h"] = "void headerFunc();";
  Inputs.Code = R"cpp(
    #include "header.h"
    void mainFunc();
  )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(named("mainFunc")));
}

// Decl from macro expanded into the main file is a root.
TEST_F(RecordASTTest, Macros) {
  Inputs.ExtraFiles["header.h"] = "#define X void x();";
  Inputs.Code = R"cpp(
    #include "header.h"
    X
  )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots, testing::ElementsAre(named("x")));
}

// Decl from template instantiation is filtered out from roots.
TEST_F(RecordASTTest, ImplicitTemplates) {
  Inputs.ExtraFiles["dispatch.h"] = R"cpp(
  struct A {
    static constexpr int value = 1;
  };
  template <class Getter>
  int dispatch() {
    return Getter::template get<A>();
  }
  )cpp";
  Inputs.Code = R"cpp(
  #include "dispatch.h"  
  struct MyGetter {
    template <class T> static int get() { return T::value; }
  };
  int v = dispatch<MyGetter>();
  )cpp";
  auto AST = build();
  EXPECT_THAT(Recorded.Roots,
              testing::ElementsAre(named("MyGetter"), named("v")));
}

class RecordPPTest : public ::testing::Test {
protected:
  TestInputs Inputs;
  RecordedPP Recorded;

  RecordPPTest() {
    struct RecordAction : public PreprocessOnlyAction {
      RecordedPP &Out;
      RecordAction(RecordedPP &Out) : Out(Out) {}

      void ExecuteAction() override {
        auto &PP = getCompilerInstance().getPreprocessor();
        PP.addPPCallbacks(Out.record(PP));
        PreprocessOnlyAction::ExecuteAction();
      }
    };
    Inputs.MakeAction = [this] {
      return std::make_unique<RecordAction>(Recorded);
    };
  }

  TestAST build() { return TestAST(Inputs); }
};

// Matches an Include with a particular spelling.
MATCHER_P(spelled, S, "") { return arg.Spelled == S; }

TEST_F(RecordPPTest, CapturesIncludes) {
  llvm::Annotations MainFile(R"cpp(
    $H^#include "./header.h"
    $M^#include <missing.h>
  )cpp");
  Inputs.Code = MainFile.code();
  Inputs.ExtraFiles["header.h"] = "";
  Inputs.ErrorOK = true; // missing header
  auto AST = build();

  ASSERT_THAT(
      Recorded.Includes.all(),
      testing::ElementsAre(spelled("./header.h"), spelled("missing.h")));

  auto &H = Recorded.Includes.all().front();
  EXPECT_EQ(H.Line, 2u);
  EXPECT_EQ(H.HashLocation,
            AST.sourceManager().getComposedLoc(
                AST.sourceManager().getMainFileID(), MainFile.point("H")));
  EXPECT_EQ(H.Resolved, *AST.fileManager().getOptionalFileRef("header.h"));
  EXPECT_FALSE(H.Angled);

  auto &M = Recorded.Includes.all().back();
  EXPECT_EQ(M.Line, 3u);
  EXPECT_EQ(M.HashLocation,
            AST.sourceManager().getComposedLoc(
                AST.sourceManager().getMainFileID(), MainFile.point("M")));
  EXPECT_EQ(M.Resolved, std::nullopt);
  EXPECT_TRUE(M.Angled);
}

TEST_F(RecordPPTest, CapturesMacroRefs) {
  llvm::Annotations Header(R"cpp(
    #define $def^X 1

    // Refs, but not in main file.
    #define Y X
    int one = X;
  )cpp");
  llvm::Annotations MainFile(R"cpp(
    #define EARLY X // not a ref, no definition
    #include "header.h"
    #define LATE ^X
    #define LATE2 ^X // a ref even if not expanded

    int uno = ^X;
    int jeden = $exp^LATE; // a ref in LATE's expansion

    #define IDENT(X) X // not a ref, shadowed
    int eins = IDENT(^X);

    #undef ^X
    // Not refs, rather a new macro with the same name.
    #define X 2
    int two = X;
  )cpp");
  Inputs.Code = MainFile.code();
  Inputs.ExtraFiles["header.h"] = Header.code();
  auto AST = build();
  const auto &SM = AST.sourceManager();

  SourceLocation Def = SM.getComposedLoc(
      SM.translateFile(AST.fileManager().getFile("header.h").get()),
      Header.point("def"));
  ASSERT_THAT(Recorded.MacroReferences, Not(IsEmpty()));
  Symbol OrigX = Recorded.MacroReferences.front().Target;
  EXPECT_EQ("X", OrigX.macro().Name->getName());
  EXPECT_EQ(Def, OrigX.macro().Definition);

  std::vector<unsigned> RefOffsets;
  std::vector<unsigned> ExpOffsets; // Expansion locs of refs in macro locs.
  for (const auto &Ref : Recorded.MacroReferences) {
    if (Ref.Target == OrigX) {
      auto [FID, Off] = SM.getDecomposedLoc(Ref.RefLocation);
      if (FID == SM.getMainFileID()) {
        RefOffsets.push_back(Off);
      } else if (Ref.RefLocation.isMacroID() &&
                 SM.isWrittenInMainFile(SM.getExpansionLoc(Ref.RefLocation))) {
        ExpOffsets.push_back(
            SM.getDecomposedExpansionLoc(Ref.RefLocation).second);
      } else {
        ADD_FAILURE() << Ref.RefLocation.printToString(SM);
      }
    }
  }
  EXPECT_THAT(RefOffsets, ElementsAreArray(MainFile.points()));
  EXPECT_THAT(ExpOffsets, ElementsAreArray(MainFile.points("exp")));
}

TEST_F(RecordPPTest, CapturesConditionalMacroRefs) {
  llvm::Annotations MainFile(R"cpp(
    #define X 1

    #ifdef ^X
    #endif

    #if defined(^X)
    #endif

    #ifndef ^X
    #endif

    #ifdef Y
    #elifdef ^X
    #endif

    #ifndef ^X
    #elifndef ^X
    #endif
  )cpp");

  Inputs.Code = MainFile.code();
  Inputs.ExtraArgs.push_back("-std=c++2b");
  auto AST = build();

  std::vector<unsigned> RefOffsets;
  SourceManager &SM = AST.sourceManager();
  for (const auto &Ref : Recorded.MacroReferences) {
    auto [FID, Off] = SM.getDecomposedLoc(Ref.RefLocation);
    ASSERT_EQ(FID, SM.getMainFileID());
    EXPECT_EQ(Ref.RT, RefType::Ambiguous);
    EXPECT_EQ("X", Ref.Target.macro().Name->getName());
    RefOffsets.push_back(Off);
  }
  EXPECT_THAT(RefOffsets, ElementsAreArray(MainFile.points()));
}

class PragmaIncludeTest : public ::testing::Test {
protected:
  // We don't build an AST, we just run a preprocessor action!
  TestInputs Inputs;
  PragmaIncludes PI;

  PragmaIncludeTest() {
    Inputs.MakeAction = [this] {
      struct Hook : public PreprocessOnlyAction {
      public:
        Hook(PragmaIncludes *Out) : Out(Out) {}
        bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
          Out->record(CI);
          return true;
        }
        PragmaIncludes *Out;
      };
      return std::make_unique<Hook>(&PI);
    };
  }

  TestAST build() { return TestAST(Inputs); }

  void createEmptyFiles(llvm::ArrayRef<StringRef> FileNames) {
    for (llvm::StringRef File : FileNames)
      Inputs.ExtraFiles[File] = "#pragma once";
  }
};

TEST_F(PragmaIncludeTest, IWYUKeep) {
  Inputs.Code = R"cpp(
    #include "keep1.h" // IWYU pragma: keep
    #include "keep2.h" /* IWYU pragma: keep */

    #include "export1.h" // IWYU pragma: export
    // IWYU pragma: begin_exports
    #include "export2.h"
    #include "export3.h"
    // IWYU pragma: end_exports

    #include "normal.h"

    // IWYU pragma: begin_keep
    #include "keep3.h"
    // IWYU pragma: end_keep

    // IWYU pragma: begin_keep
    #include "keep4.h"
    // IWYU pragma: begin_keep
    #include "keep5.h"
    // IWYU pragma: end_keep
    #include "keep6.h"
    // IWYU pragma: end_keep
    #include <vector>
    #include <map> // IWYU pragma: keep
    #include <set> // IWYU pragma: export
  )cpp";
  createEmptyFiles({"keep1.h", "keep2.h", "keep3.h", "keep4.h", "keep5.h",
                    "keep6.h", "export1.h", "export2.h", "export3.h",
                    "normal.h", "std/vector", "std/map", "std/set"});

  Inputs.ExtraArgs.push_back("-isystemstd");
  TestAST Processed = build();
  auto &FM = Processed.fileManager();

  EXPECT_FALSE(PI.shouldKeep(FM.getFile("normal.h").get()));
  EXPECT_FALSE(PI.shouldKeep(FM.getFile("std/vector").get()));

  // Keep
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep1.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep2.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep3.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep4.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep5.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("keep6.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("std/map").get()));

  // Exports
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("export1.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("export2.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("export3.h").get()));
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("std/set").get()));
}

TEST_F(PragmaIncludeTest, IWYUPrivate) {
  Inputs.Code = R"cpp(
    #include "public.h"
  )cpp";
  Inputs.ExtraFiles["public.h"] = R"cpp(
    #include "private.h"
    #include "private2.h"
  )cpp";
  Inputs.ExtraFiles["private.h"] = R"cpp(
    // IWYU pragma: private, include "public2.h"
  )cpp";
  Inputs.ExtraFiles["private2.h"] = R"cpp(
    // IWYU pragma: private
  )cpp";
  TestAST Processed = build();
  auto PrivateFE = Processed.fileManager().getFile("private.h");
  assert(PrivateFE);
  EXPECT_TRUE(PI.isPrivate(PrivateFE.get()));
  EXPECT_EQ(PI.getPublic(PrivateFE.get()), "\"public2.h\"");

  auto PublicFE = Processed.fileManager().getFile("public.h");
  assert(PublicFE);
  EXPECT_EQ(PI.getPublic(PublicFE.get()), ""); // no mapping.
  EXPECT_FALSE(PI.isPrivate(PublicFE.get()));

  auto Private2FE = Processed.fileManager().getFile("private2.h");
  assert(Private2FE);
  EXPECT_TRUE(PI.isPrivate(Private2FE.get()));
}

TEST_F(PragmaIncludeTest, IWYUExport) {
  Inputs.Code = R"cpp(// Line 1
    #include "export1.h"
    #include "export2.h"
  )cpp";
  Inputs.ExtraFiles["export1.h"] = R"cpp(
    #include "private.h" // IWYU pragma: export
  )cpp";
  Inputs.ExtraFiles["export2.h"] = R"cpp(
    #include "export3.h"
  )cpp";
  Inputs.ExtraFiles["export3.h"] = R"cpp(
    #include "private.h" // IWYU pragma: export
  )cpp";
  Inputs.ExtraFiles["private.h"] = "";
  TestAST Processed = build();
  const auto &SM = Processed.sourceManager();
  auto &FM = Processed.fileManager();

  EXPECT_THAT(PI.getExporters(FM.getFile("private.h").get(), FM),
              testing::UnorderedElementsAre(FileNamed("export1.h"),
                                            FileNamed("export3.h")));

  EXPECT_TRUE(PI.getExporters(FM.getFile("export1.h").get(), FM).empty());
  EXPECT_TRUE(PI.getExporters(FM.getFile("export2.h").get(), FM).empty());
  EXPECT_TRUE(PI.getExporters(FM.getFile("export3.h").get(), FM).empty());
  EXPECT_TRUE(
      PI.getExporters(SM.getFileEntryForID(SM.getMainFileID()), FM).empty());
}

TEST_F(PragmaIncludeTest, IWYUExportForStandardHeaders) {
  Inputs.Code = R"cpp(
    #include "export.h"
  )cpp";
  Inputs.ExtraFiles["export.h"] = R"cpp(
    #include <string> // IWYU pragma: export
  )cpp";
  Inputs.ExtraFiles["string"] = "";
  Inputs.ExtraArgs = {"-isystem."};
  TestAST Processed = build();
  auto &FM = Processed.fileManager();
  EXPECT_THAT(PI.getExporters(*tooling::stdlib::Header::named("<string>"), FM),
              testing::UnorderedElementsAre(FileNamed("export.h")));
}

TEST_F(PragmaIncludeTest, IWYUExportBlock) {
  Inputs.Code = R"cpp(// Line 1
   #include "normal.h"
  )cpp";
  Inputs.ExtraFiles["normal.h"] = R"cpp(
    #include "foo.h"

    // IWYU pragma: begin_exports
    #include "export1.h"
    #include "private1.h"
    // IWYU pragma: end_exports
  )cpp";
  Inputs.ExtraFiles["export1.h"] = R"cpp(
    // IWYU pragma: begin_exports
    #include "private1.h"
    #include "private2.h"
    // IWYU pragma: end_exports

    #include "bar.h"
    #include "private3.h" // IWYU pragma: export
  )cpp";
  createEmptyFiles(
      {"private1.h", "private2.h", "private3.h", "foo.h", "bar.h"});
  TestAST Processed = build();
  auto &FM = Processed.fileManager();

  EXPECT_THAT(PI.getExporters(FM.getFile("private1.h").get(), FM),
              testing::UnorderedElementsAre(FileNamed("export1.h"),
                                            FileNamed("normal.h")));
  EXPECT_THAT(PI.getExporters(FM.getFile("private2.h").get(), FM),
              testing::UnorderedElementsAre(FileNamed("export1.h")));
  EXPECT_THAT(PI.getExporters(FM.getFile("private3.h").get(), FM),
              testing::UnorderedElementsAre(FileNamed("export1.h")));

  EXPECT_TRUE(PI.getExporters(FM.getFile("foo.h").get(), FM).empty());
  EXPECT_TRUE(PI.getExporters(FM.getFile("bar.h").get(), FM).empty());
}

TEST_F(PragmaIncludeTest, SelfContained) {
  Inputs.Code = R"cpp(
  #include "guarded.h"

  #include "unguarded.h"
  )cpp";
  Inputs.ExtraFiles["guarded.h"] = R"cpp(
  #pragma once
  )cpp";
  Inputs.ExtraFiles["unguarded.h"] = "";
  TestAST Processed = build();
  auto &FM = Processed.fileManager();
  EXPECT_TRUE(PI.isSelfContained(FM.getFile("guarded.h").get()));
  EXPECT_FALSE(PI.isSelfContained(FM.getFile("unguarded.h").get()));
}

TEST_F(PragmaIncludeTest, AlwaysKeep) {
  Inputs.Code = R"cpp(
  #include "always_keep.h"
  #include "usual.h"
  )cpp";
  Inputs.ExtraFiles["always_keep.h"] = R"cpp(
  #pragma once
  // IWYU pragma: always_keep
  )cpp";
  Inputs.ExtraFiles["usual.h"] = "#pragma once";
  TestAST Processed = build();
  auto &FM = Processed.fileManager();
  EXPECT_TRUE(PI.shouldKeep(FM.getFile("always_keep.h").get()));
  EXPECT_FALSE(PI.shouldKeep(FM.getFile("usual.h").get()));
}

TEST_F(PragmaIncludeTest, ExportInUnnamedBuffer) {
  llvm::StringLiteral Filename = "test.cpp";
  auto Code = R"cpp(#include "exporter.h")cpp";
  Inputs.ExtraFiles["exporter.h"] = R"cpp(
  #pragma once
  #include "foo.h" // IWYU pragma: export
  )cpp";
  Inputs.ExtraFiles["foo.h"] = "";

  auto Clang = std::make_unique<CompilerInstance>(
      std::make_shared<PCHContainerOperations>());
  Clang->createDiagnostics();

  Clang->setInvocation(std::make_unique<CompilerInvocation>());
  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(
      Clang->getInvocation(), {Filename.data()}, Clang->getDiagnostics(),
      "clang"));

  // Create unnamed memory buffers for all the files.
  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->addFile(Filename, /*ModificationTime=*/0,
               llvm::MemoryBuffer::getMemBufferCopy(Code, /*BufferName=*/""));
  for (const auto &Extra : Inputs.ExtraFiles)
    VFS->addFile(Extra.getKey(), /*ModificationTime=*/0,
                 llvm::MemoryBuffer::getMemBufferCopy(Extra.getValue(),
                                                      /*BufferName=*/""));
  auto *FM = Clang->createFileManager(VFS);
  ASSERT_TRUE(Clang->ExecuteAction(*Inputs.MakeAction()));
  EXPECT_THAT(
      PI.getExporters(llvm::cantFail(FM->getFileRef("foo.h")), *FM),
      testing::ElementsAre(llvm::cantFail(FM->getFileRef("exporter.h"))));
}
} // namespace
} // namespace clang::include_cleaner
