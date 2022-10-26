//===-- RecordTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;
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
    $M^#include "missing.h"
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
  EXPECT_EQ(H.Resolved, AST.fileManager().getFile("header.h").get());

  auto &M = Recorded.Includes.all().back();
  EXPECT_EQ(M.Line, 3u);
  EXPECT_EQ(M.HashLocation,
            AST.sourceManager().getComposedLoc(
                AST.sourceManager().getMainFileID(), MainFile.point("M")));
  EXPECT_EQ(M.Resolved, nullptr);
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
  Symbol OrigX = Recorded.MacroReferences.front().Symbol;
  EXPECT_EQ("X", OrigX.macro().Name->getName());
  EXPECT_EQ(Def, OrigX.macro().Definition);

  std::vector<unsigned> RefOffsets;
  std::vector<unsigned> ExpOffsets; // Expansion locs of refs in macro locs.
  std::vector<SourceLocation> RefMacroLocs;
  for (const auto &Ref : Recorded.MacroReferences) {
    if (Ref.Symbol == OrigX) {
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

// Matches an Include* on the specified line;
MATCHER_P(line, N, "") { return arg->Line == (unsigned)N; }

TEST(RecordedIncludesTest, Match) {
  // We're using synthetic data, but need a FileManager to obtain FileEntry*s.
  // Ensure it doesn't do any actual IO.
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager FM(FileSystemOptions{});
  const FileEntry *A = FM.getVirtualFile("/path/a", /*Size=*/0, time_t{});
  const FileEntry *B = FM.getVirtualFile("/path/b", /*Size=*/0, time_t{});

  RecordedPP::RecordedIncludes Includes;
  Includes.add(Include{"a", A, SourceLocation(), 1});
  Includes.add(Include{"a2", A, SourceLocation(), 2});
  Includes.add(Include{"b", B, SourceLocation(), 3});
  Includes.add(Include{"vector", B, SourceLocation(), 4});
  Includes.add(Include{"vector", B, SourceLocation(), 5});
  Includes.add(Include{"missing", nullptr, SourceLocation(), 6});

  EXPECT_THAT(Includes.match(A), ElementsAre(line(1), line(2)));
  EXPECT_THAT(Includes.match(B), ElementsAre(line(3), line(4), line(5)));
  EXPECT_THAT(Includes.match(*tooling::stdlib::Header::named("<vector>")),
              ElementsAre(line(4), line(5)));
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
};

TEST_F(PragmaIncludeTest, IWYUKeep) {
  Inputs.Code = R"cpp(// Line 1
    #include "keep1.h" // IWYU pragma: keep
    #include "keep2.h" /* IWYU pragma: keep */
    #include "normal.h"
  )cpp";
  Inputs.ExtraFiles["keep1.h"] = Inputs.ExtraFiles["keep2.h"] =
      Inputs.ExtraFiles["normal.h"] = "";

  TestAST Processed = build();
  EXPECT_FALSE(PI.shouldKeep(1));
  EXPECT_TRUE(PI.shouldKeep(2));
  EXPECT_TRUE(PI.shouldKeep(3));
  EXPECT_FALSE(PI.shouldKeep(4));
}

TEST_F(PragmaIncludeTest, IWYUPrivate) {
  Inputs.Code = R"cpp(
    #include "public.h"
  )cpp";
  Inputs.ExtraFiles["public.h"] = "#include \"private.h\"";
  Inputs.ExtraFiles["private.h"] = R"cpp(
    // IWYU pragma: private, include "public2.h"
    class Private {};
  )cpp";
  TestAST Processed = build();
  auto PrivateFE = Processed.fileManager().getFile("private.h");
  assert(PrivateFE);
  EXPECT_EQ(PI.getPublic(PrivateFE.get()), "\"public2.h\"");
  auto PublicFE = Processed.fileManager().getFile("public.h");
  assert(PublicFE);
  EXPECT_EQ(PI.getPublic(PublicFE.get()), ""); // no mapping.
}

} // namespace
} // namespace clang::include_cleaner
