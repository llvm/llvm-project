//=== unittests/Tooling/BoundsSafetyBringupMissingChecks.cpp =================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This test is really a `Frontend` test but it's in `Tooling` so we can use
// `clang::tooling::runToolOnCodeWithArgs` to make writing the unit test much
// easier.
//
// This test verifies LangOptions gets set appropriately based on the provided
// `-fbounds-safety-bringup-missing-checks` flags.
//
//===----------------------------------------------------------------------===//
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::tooling;

// FIXME: These functions are a workaround for
// `clang::tooling::runToolOnCodeWithArgs` not returning false when handling
// unrecognized driver flags (rdar://138379948). This isn't the right fix.
// The radar explains the correct way to fix this.
static std::vector<std::string>
getSyntaxOnlyToolArgs(const Twine &ToolName,
                      const std::vector<std::string> &ExtraArgs,
                      StringRef FileName) {
  std::vector<std::string> Args;
  Args.push_back(ToolName.str());
  Args.push_back("-fsyntax-only");
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  Args.push_back(FileName.str());
  return Args;
}

static bool
runToolOnCodeWithArgs(std::unique_ptr<FrontendAction> ToolAction,
                      const Twine &Code,
                      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                      const std::vector<std::string> &Args,
                      const Twine &FileName, const Twine &ToolName,
                      std::shared_ptr<PCHContainerOperations> PCHContainerOps =
                          std::make_shared<PCHContainerOperations>()) {
  SmallString<16> FileNameStorage;
  StringRef FileNameRef = FileName.toNullTerminatedStringRef(FileNameStorage);

  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), VFS));
  ArgumentsAdjuster Adjuster = getClangStripDependencyFileAdjuster();
  ToolInvocation Invocation(
      getSyntaxOnlyToolArgs(ToolName, Adjuster(Args, FileNameRef), FileNameRef),
      std::move(ToolAction), Files.get(), std::move(PCHContainerOps));

  // Workaround rdar://138379948. Note this isn't the right way to fix the bug
  // but it's good enough for the purposes of this test.
  std::vector<const char *> ArgsCC;
  for (const auto &Arg : Args)
    ArgsCC.push_back(Arg.c_str());
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      clang::CreateAndPopulateDiagOpts(ArgsCC);
  auto TDP =
      std::make_unique<TextDiagnosticPrinter>(llvm::errs(), DiagOpts.get());
  Invocation.setDiagnosticConsumer(TDP.get());

  return Invocation.run() && TDP->getNumErrors() == 0;
}

static bool runToolOnCodeWithArgs(
    std::unique_ptr<FrontendAction> ToolAction, const Twine &Code,
    const std::vector<std::string> &Args, const Twine &FileName,
    const Twine &ToolName = "clang-tool",
    std::shared_ptr<PCHContainerOperations> PCHContainerOps =
        std::make_shared<PCHContainerOperations>(),
    const FileContentMappings &VirtualMappedFiles = FileContentMappings()) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);

  SmallString<1024> CodeStorage;
  InMemoryFileSystem->addFile(FileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(
                                  Code.toNullTerminatedStringRef(CodeStorage)));

  for (auto &FilenameWithContent : VirtualMappedFiles) {
    InMemoryFileSystem->addFile(
        FilenameWithContent.first, 0,
        llvm::MemoryBuffer::getMemBuffer(FilenameWithContent.second));
  }

  return ::runToolOnCodeWithArgs(std::move(ToolAction), Code, OverlayFileSystem,
                                 Args, FileName, ToolName);
}

using LangOptionsTestFn = std::function<void(LangOptions &)>;

bool runOnToolAndCheckLangOptions(const std::vector<std::string> &Args,
                                  LangOptionsTestFn Handler) {
  class CheckLangOptions : public clang::InitOnlyAction {
    LangOptionsTestFn Handler;

  public:
    CheckLangOptions(LangOptionsTestFn Handler) : Handler(Handler) {}
    bool BeginSourceFileAction(CompilerInstance &CI) override {
      auto &LangOpts = getCompilerInstance().getLangOpts();
      Handler(LangOpts);
      return true;
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
      return std::make_unique<ASTConsumer>();
    }
  };

  auto Action = std::make_unique<CheckLangOptions>(Handler);
  // FIXME: Workaround invalid driver args not causing false to be returned
  // (rdar://138379948) by calling a modified version of
  // `runToolOnCodeWithArgs`.
  bool Result = /*clang::tooling*/ ::runToolOnCodeWithArgs(
      std::move(Action), /*Code=*/"", Args, "test.c");

  return Result;
}

struct ChkDesc {
  const char *arg;
  unsigned Mask;
  unsigned batch;
};

enum BatchedChecks { BATCH_ZERO = 0 };

ChkDesc CheckKinds[] = {
    {"access_size", LangOptions::BS_CHK_AccessSize, BATCH_ZERO},
    {"indirect_count_update", LangOptions::BS_CHK_IndirectCountUpdate,
     BATCH_ZERO},
    {"return_size", LangOptions::BS_CHK_ReturnSize, BATCH_ZERO},
    {"ended_by_lower_bound", LangOptions::BS_CHK_EndedByLowerBound, BATCH_ZERO},
    {"compound_literal_init", LangOptions::BS_CHK_CompoundLiteralInit,
     BATCH_ZERO},
    {"libc_attributes", LangOptions::BS_CHK_LibCAttributes, BATCH_ZERO},
    {"array_subscript_agg", LangOptions::BS_CHK_ArraySubscriptAgg, BATCH_ZERO}};
const size_t NumChkDescs = sizeof(CheckKinds) / sizeof(CheckKinds[0]);
static_assert(NumChkDescs == 7, "Unexpected value");

// Check that `Pairs` is in sync with the `BoundsSafetyNewChecksMask`
// enum and that the special "all" group contains all the checks
TEST(BoundsSafetyBringUpMissingChecks, ChkPairInSync) {
  unsigned ComputedMask = 0;
  static_assert(LangOptions::BS_CHK_None == 0, "expected 0");
  for (size_t Idx = 0; Idx < NumChkDescs; ++Idx) {
    ComputedMask |= CheckKinds[Idx].Mask;
  }
  ASSERT_EQ(ComputedMask,
            LangOptions::getBoundsSafetyNewChecksMaskForGroup("all"));
}

// Check that the "batch_0" group contains the right checks
TEST(BoundsSafetyBringUpMissingChecks, ChkPairInSyncBatch0) {
  unsigned ComputedMask = 0;
  static_assert(LangOptions::BS_CHK_None == 0, "expected 0");
  for (size_t Idx = 0; Idx < NumChkDescs; ++Idx) {
    if (CheckKinds[Idx].batch == BATCH_ZERO)
      ComputedMask |= CheckKinds[Idx].Mask;
  }
  ASSERT_EQ(ComputedMask,
            LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0"));
}

TEST(BoundsSafetyBringUpMissingChecks, ChkPairValidMask) {
  unsigned SeenBits = 0;
  static_assert(LangOptions::BS_CHK_None == 0, "expected 0");
  for (size_t Idx = 0; Idx < NumChkDescs; ++Idx) {
    unsigned CurrentMask = CheckKinds[Idx].Mask;
    EXPECT_EQ(__builtin_popcount(CurrentMask), 1); // Check is a power of 2
    EXPECT_EQ(SeenBits & CurrentMask,
              0U); // Doesn't overlap with a previously seen value
    SeenBits |= CurrentMask;
    EXPECT_NE(SeenBits, 0U);
  }
  ASSERT_EQ(SeenBits, LangOptions::getBoundsSafetyNewChecksMaskForGroup("all"));
}

#if 1
#define NEED_CC1_ARG
#else
#define NEED_CC1_ARG "-Xclang",
#endif

// =============================================================================
// Default behavior
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, DefaultWithBoundsSafety) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety"}, [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getDefaultBoundsSafetyNewChecksMask());
      });
  ASSERT_TRUE(Result);
}

TEST(BoundsSafetyBringUpMissingChecks, DefaultWithoutBoundsSafety) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fno-bounds-safety"}, [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getDefaultBoundsSafetyNewChecksMask());
      });
  ASSERT_TRUE(Result);
  bool Result2 = runOnToolAndCheckLangOptions({""}, [](LangOptions &LO) {
    EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
              LangOptions::getDefaultBoundsSafetyNewChecksMask());
  });
  ASSERT_TRUE(Result2);
}

TEST(BoundsSafetyBringUpMissingChecks,
     DefaultAllDisabledExperimentalBoundsSafetyAttributes) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fexperimental-bounds-safety-attributes"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getDefaultBoundsSafetyNewChecksMask());
      });
  ASSERT_TRUE(Result);
}

// =============================================================================
// All checks enabled
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, all_eq) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety",
       NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks=all"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getBoundsSafetyNewChecksMaskForGroup("all"));

        for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(CheckKinds[ChkIdx].Mask));
        }
      });
  ASSERT_TRUE(Result);
}

TEST(BoundsSafetyBringUpMissingChecks, all) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety",
       NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getBoundsSafetyNewChecksMaskForGroup("all"));

        for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(CheckKinds[ChkIdx].Mask));
        }
      });
  ASSERT_TRUE(Result);
}

// =============================================================================
// batch_0 checks enabled
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, batch_0) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety",
       NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks=batch_0"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0"));

        for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
          if (CheckKinds[ChkIdx].batch == BATCH_ZERO)
            EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(CheckKinds[ChkIdx].Mask));
        }
      });
  ASSERT_TRUE(Result);
}

// =============================================================================
// Specify just one check
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, only_one_check) {
  for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
    ChkDesc Chk = CheckKinds[ChkIdx];
    std::vector<std::string> Args = {NEED_CC1_ARG "-fbounds-safety",
                                     NEED_CC1_ARG
                                     "-fbounds-safety-bringup-missing-checks="};
    Args[Args.size() - 1].append(Chk.arg);

    bool Result = runOnToolAndCheckLangOptions(Args, [&Chk](LangOptions &LO) {
      EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks, Chk.Mask);

      // Check helper
      for (size_t OtherChkIdx = 0; OtherChkIdx < NumChkDescs; ++OtherChkIdx) {
        ChkDesc OtherChk = CheckKinds[OtherChkIdx];
        if (OtherChk.Mask == Chk.Mask) {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        } else {
          EXPECT_FALSE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        }
      }
    });
    ASSERT_TRUE(Result);
  }
}

// =============================================================================
// Pairs of checks
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, all_pairs) {
  // Construct all distinct pairs and test
  for (size_t firstIdx = 0; firstIdx < NumChkDescs; ++firstIdx) {
    for (size_t secondIdx = firstIdx + 1; secondIdx < NumChkDescs;
         ++secondIdx) {
      ASSERT_NE(firstIdx, secondIdx);
      ChkDesc First = CheckKinds[firstIdx];
      auto Second = CheckKinds[secondIdx];
      std::vector<std::string> Args = {
          NEED_CC1_ARG "-fbounds-safety",
          NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks="};
      Args[Args.size() - 1].append(First.arg);
      Args[Args.size() - 1].append(",");
      Args[Args.size() - 1].append(Second.arg);
      ASSERT_NE(First.Mask, Second.Mask);

      bool Result = runOnToolAndCheckLangOptions(
          Args, [&First, &Second](LangOptions &LO) {
            unsigned ExpectedMask = First.Mask | Second.Mask;
            EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks, ExpectedMask);

            // Check helper
            for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
              unsigned ChkToTest = CheckKinds[ChkIdx].Mask;
              if (ChkToTest == First.Mask || ChkToTest == Second.Mask) {
                EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(ChkToTest));
              } else {
                EXPECT_FALSE(LO.hasNewBoundsSafetyCheck(ChkToTest));
              }
            }
          });
      ASSERT_TRUE(Result);
    }
  }
}

// =============================================================================
// All checks with one removed
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, all_with_one_removed) {
  for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
    ChkDesc Chk = CheckKinds[ChkIdx];
    std::vector<std::string> Args = {
        NEED_CC1_ARG "-fbounds-safety",
        NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks=all",
        NEED_CC1_ARG "-fno-bounds-safety-bringup-missing-checks="};
    Args[Args.size() - 1].append(Chk.arg);

    bool Result = runOnToolAndCheckLangOptions(Args, [&Chk](LangOptions &LO) {
      unsigned ExpectedMask =
          LangOptions::getBoundsSafetyNewChecksMaskForGroup("all") &
          (~Chk.Mask);
      EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks, ExpectedMask);

      // Check helper
      for (size_t OtherChkIdx = 0; OtherChkIdx < NumChkDescs; ++OtherChkIdx) {
        ChkDesc OtherChk = CheckKinds[OtherChkIdx];
        if (OtherChk.Mask == Chk.Mask) {
          EXPECT_FALSE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        } else {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        }
      }
    });
    ASSERT_TRUE(Result);
  }
}

TEST(BoundsSafetyBringUpMissingChecks, batch_0_with_one_removed) {
  for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
    ChkDesc Chk = CheckKinds[ChkIdx];
    if (Chk.batch != 0)
      continue;

    std::vector<std::string> Args = {
        NEED_CC1_ARG "-fbounds-safety",
        NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks=batch_0",
        NEED_CC1_ARG "-fno-bounds-safety-bringup-missing-checks="};
    Args[Args.size() - 1].append(Chk.arg);

    bool Result = runOnToolAndCheckLangOptions(Args, [&Chk](LangOptions &LO) {
      unsigned ExpectedMask =
          LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0") &
          (~Chk.Mask);
      EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks, ExpectedMask);

      // Check helper
      for (size_t OtherChkIdx = 0; OtherChkIdx < NumChkDescs; ++OtherChkIdx) {
        ChkDesc OtherChk = CheckKinds[OtherChkIdx];
        if (OtherChk.batch != BATCH_ZERO)
          continue;
        if (OtherChk.Mask == Chk.Mask) {
          EXPECT_FALSE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        } else {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        }
      }
    });
    ASSERT_TRUE(Result);
  }
}

// =============================================================================
// No checks
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, all_disabled) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety",
       NEED_CC1_ARG "-fno-bounds-safety-bringup-missing-checks"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::BS_CHK_None);
      });
  ASSERT_TRUE(Result);
}

TEST(BoundsSafetyBringUpMissingChecks, all_enable_then_disable) {
  bool Result = runOnToolAndCheckLangOptions(
      {NEED_CC1_ARG "-fbounds-safety",
       NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks",
       NEED_CC1_ARG "-fno-bounds-safety-bringup-missing-checks"},
      [](LangOptions &LO) {
        EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks,
                  LangOptions::BS_CHK_None);
      });
  ASSERT_TRUE(Result);
}

// =============================================================================
// Explicitly disable all checks then enable one
// =============================================================================

TEST(BoundsSafetyBringUpMissingChecks, all_disabled_then_enable_one) {
  for (size_t ChkIdx = 0; ChkIdx < NumChkDescs; ++ChkIdx) {
    ChkDesc Chk = CheckKinds[ChkIdx];
    std::vector<std::string> Args = {
        NEED_CC1_ARG "-fbounds-safety",
        NEED_CC1_ARG "-fno-bounds-safety-bringup-missing-checks",
        NEED_CC1_ARG "-fbounds-safety-bringup-missing-checks="};
    Args[Args.size() - 1].append(Chk.arg);

    bool Result = runOnToolAndCheckLangOptions(Args, [&Chk](LangOptions &LO) {
      EXPECT_EQ(LO.BoundsSafetyBringUpMissingChecks, Chk.Mask);

      // Check helper
      for (size_t OtherChkIdx = 0; OtherChkIdx < NumChkDescs; ++OtherChkIdx) {
        ChkDesc OtherChk = CheckKinds[OtherChkIdx];
        if (OtherChk.Mask == Chk.Mask) {
          EXPECT_TRUE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        } else {
          EXPECT_FALSE(LO.hasNewBoundsSafetyCheck(OtherChk.Mask));
        }
      }
    });
    ASSERT_TRUE(Result);
  }
}

