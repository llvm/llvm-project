//===- unittests/Basic/SourceManagerTest.cpp ------ SourceManager tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"
#include <cstddef>

using namespace clang;

namespace clang {
class SourceManagerTestHelper {
public:
  static FileID makeFileID(int ID) { return FileID::get(ID); }
};
} // namespace clang

namespace {

// The test fixture.
class SourceManagerTest : public ::testing::Test {
protected:
  SourceManagerTest()
      : FileMgr(FileMgrOpts),
        Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

TEST_F(SourceManagerTest, isInMemoryBuffersNoSourceLocationInfo) {
  // Check for invalid source location for each method
  SourceLocation LocEmpty;
  bool isWrittenInBuiltInFileFalse = SourceMgr.isWrittenInBuiltinFile(LocEmpty);
  bool isWrittenInCommandLineFileFalse =
      SourceMgr.isWrittenInCommandLineFile(LocEmpty);
  bool isWrittenInScratchSpaceFalse =
      SourceMgr.isWrittenInScratchSpace(LocEmpty);

  EXPECT_FALSE(isWrittenInBuiltInFileFalse);
  EXPECT_FALSE(isWrittenInCommandLineFileFalse);
  EXPECT_FALSE(isWrittenInScratchSpaceFalse);

  // Check for valid source location per filename for each method
  const char *Source = "int x";

  std::unique_ptr<llvm::MemoryBuffer> BuiltInBuf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileEntryRef BuiltInFile =
      FileMgr.getVirtualFileRef("<built-in>", BuiltInBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(BuiltInFile, std::move(BuiltInBuf));
  FileID BuiltInFileID =
      SourceMgr.getOrCreateFileID(BuiltInFile, SrcMgr::C_User);
  SourceMgr.setMainFileID(BuiltInFileID);
  SourceLocation LocBuiltIn =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());
  bool isWrittenInBuiltInFileTrue =
      SourceMgr.isWrittenInBuiltinFile(LocBuiltIn);

  std::unique_ptr<llvm::MemoryBuffer> CommandLineBuf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileEntryRef CommandLineFile = FileMgr.getVirtualFileRef(
      "<command line>", CommandLineBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(CommandLineFile, std::move(CommandLineBuf));
  FileID CommandLineFileID =
      SourceMgr.getOrCreateFileID(CommandLineFile, SrcMgr::C_User);
  SourceMgr.setMainFileID(CommandLineFileID);
  SourceLocation LocCommandLine =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());
  bool isWrittenInCommandLineFileTrue =
      SourceMgr.isWrittenInCommandLineFile(LocCommandLine);

  std::unique_ptr<llvm::MemoryBuffer> ScratchSpaceBuf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileEntryRef ScratchSpaceFile = FileMgr.getVirtualFileRef(
      "<scratch space>", ScratchSpaceBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(ScratchSpaceFile, std::move(ScratchSpaceBuf));
  FileID ScratchSpaceFileID =
      SourceMgr.getOrCreateFileID(ScratchSpaceFile, SrcMgr::C_User);
  SourceMgr.setMainFileID(ScratchSpaceFileID);
  SourceLocation LocScratchSpace =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());
  bool isWrittenInScratchSpaceTrue =
      SourceMgr.isWrittenInScratchSpace(LocScratchSpace);

  EXPECT_TRUE(isWrittenInBuiltInFileTrue);
  EXPECT_TRUE(isWrittenInCommandLineFileTrue);
  EXPECT_TRUE(isWrittenInScratchSpaceTrue);
}

TEST_F(SourceManagerTest, isInSystemHeader) {
  // Check for invalid source location
  SourceLocation LocEmpty;
  bool isInSystemHeaderFalse = SourceMgr.isInSystemHeader(LocEmpty);
  ASSERT_FALSE(isInSystemHeaderFalse);
}

TEST_F(SourceManagerTest, isBeforeInTranslationUnit) {
  const char *source =
    "#define M(x) [x]\n"
    "M(foo)";
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(source);
  FileID mainFileID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(mainFileID);

  HeaderSearchOptions HSOpts;
  PreprocessorOptions PPOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, &*Target);
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr, /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  PP.LexTokensUntilEOF(&toks);

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(3U, toks.size());
  ASSERT_EQ(tok::l_square, toks[0].getKind());
  ASSERT_EQ(tok::identifier, toks[1].getKind());
  ASSERT_EQ(tok::r_square, toks[2].getKind());
  
  SourceLocation lsqrLoc = toks[0].getLocation();
  SourceLocation idLoc = toks[1].getLocation();
  SourceLocation rsqrLoc = toks[2].getLocation();
  
  SourceLocation macroExpStartLoc = SourceMgr.translateLineCol(mainFileID, 2, 1);
  SourceLocation macroExpEndLoc = SourceMgr.translateLineCol(mainFileID, 2, 6);
  ASSERT_TRUE(macroExpStartLoc.isFileID());
  ASSERT_TRUE(macroExpEndLoc.isFileID());

  SmallString<32> str;
  ASSERT_EQ("M", PP.getSpelling(macroExpStartLoc, str));
  ASSERT_EQ(")", PP.getSpelling(macroExpEndLoc, str));

  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(lsqrLoc, idLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(idLoc, rsqrLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(macroExpStartLoc, idLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(idLoc, macroExpEndLoc));
}

TEST_F(SourceManagerTest, isBeforeInTranslationUnitWithTokenSplit) {
  const char *main = R"cpp(
    #define ID(X) X
    ID(
      ID(a >> b)
      c
    )
  )cpp";

  SourceMgr.setMainFileID(
      SourceMgr.createFileID(llvm::MemoryBuffer::getMemBuffer(main)));

  HeaderSearchOptions HSOpts;
  PreprocessorOptions PPOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, &*Target);
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup=*/nullptr, /*OwnsHeaderSearch=*/false);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();
  llvm::SmallString<8> Scratch;

  std::vector<Token> toks;
  PP.LexTokensUntilEOF(&toks);

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(4U, toks.size()) << "a >> b c";
  // Sanity check their order.
  for (unsigned I = 0; I < toks.size() - 1; ++I) {
    EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(toks[I].getLocation(),
                                                    toks[I + 1].getLocation()));
    EXPECT_FALSE(SourceMgr.isBeforeInTranslationUnit(toks[I + 1].getLocation(),
                                                     toks[I].getLocation()));
  }

  // Split the >> into two > tokens, as happens when parsing nested templates.
  unsigned RightShiftIndex = 1;
  SourceLocation RightShift = toks[RightShiftIndex].getLocation();
  EXPECT_EQ(">>", Lexer::getSpelling(SourceMgr.getSpellingLoc(RightShift),
                                     Scratch, SourceMgr, LangOpts));
  SourceLocation Greater1 = PP.SplitToken(RightShift, /*Length=*/1);
  SourceLocation Greater2 = RightShift.getLocWithOffset(1);
  EXPECT_TRUE(Greater1.isMacroID());
  EXPECT_EQ(">", Lexer::getSpelling(SourceMgr.getSpellingLoc(Greater1), Scratch,
                                    SourceMgr, LangOpts));
  EXPECT_EQ(">", Lexer::getSpelling(SourceMgr.getSpellingLoc(Greater2), Scratch,
                                    SourceMgr, LangOpts));
  EXPECT_EQ(SourceMgr.getImmediateExpansionRange(Greater1).getBegin(),
            RightShift);

  for (unsigned I = 0; I < toks.size(); ++I) {
    SCOPED_TRACE("Token " + std::to_string(I));
    // Right-shift is the parent of Greater1, so it compares less.
    EXPECT_EQ(
        SourceMgr.isBeforeInTranslationUnit(toks[I].getLocation(), Greater1),
        I <= RightShiftIndex);
    EXPECT_EQ(
        SourceMgr.isBeforeInTranslationUnit(toks[I].getLocation(), Greater2),
        I <= RightShiftIndex);
    EXPECT_EQ(
        SourceMgr.isBeforeInTranslationUnit(Greater1, toks[I].getLocation()),
        RightShiftIndex < I);
    EXPECT_EQ(
        SourceMgr.isBeforeInTranslationUnit(Greater2, toks[I].getLocation()),
        RightShiftIndex < I);
  }
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(Greater1, Greater2));
  EXPECT_FALSE(SourceMgr.isBeforeInTranslationUnit(Greater2, Greater1));
}

TEST_F(SourceManagerTest, getColumnNumber) {
  const char *Source =
    "int x;\n"
    "int y;";

  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileID MainFileID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(MainFileID);

  bool Invalid;

  Invalid = false;
  EXPECT_EQ(1U, SourceMgr.getColumnNumber(MainFileID, 0, &Invalid));
  EXPECT_TRUE(!Invalid);

  Invalid = false;
  EXPECT_EQ(5U, SourceMgr.getColumnNumber(MainFileID, 4, &Invalid));
  EXPECT_TRUE(!Invalid);

  Invalid = false;
  EXPECT_EQ(1U, SourceMgr.getColumnNumber(MainFileID, 7, &Invalid));
  EXPECT_TRUE(!Invalid);

  Invalid = false;
  EXPECT_EQ(5U, SourceMgr.getColumnNumber(MainFileID, 11, &Invalid));
  EXPECT_TRUE(!Invalid);

  Invalid = false;
  EXPECT_EQ(7U, SourceMgr.getColumnNumber(MainFileID, strlen(Source),
                                         &Invalid));
  EXPECT_TRUE(!Invalid);

  Invalid = false;
  SourceMgr.getColumnNumber(MainFileID, strlen(Source)+1, &Invalid);
  EXPECT_TRUE(Invalid);

  // Test invalid files
  Invalid = false;
  SourceMgr.getColumnNumber(FileID(), 0, &Invalid);
  EXPECT_TRUE(Invalid);

  Invalid = false;
  SourceMgr.getColumnNumber(FileID(), 1, &Invalid);
  EXPECT_TRUE(Invalid);

  // Test with no invalid flag.
  EXPECT_EQ(1U, SourceMgr.getColumnNumber(MainFileID, 0, nullptr));
}

TEST_F(SourceManagerTest, locationPrintTest) {
  const char *header = "#define IDENTITY(x) x\n";

  const char *Source = "int x;\n"
                       "include \"test-header.h\"\n"
                       "IDENTITY(int y);\n"
                       "int z;";

  std::unique_ptr<llvm::MemoryBuffer> HeaderBuf =
      llvm::MemoryBuffer::getMemBuffer(header);
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Source);

  FileEntryRef SourceFile =
      FileMgr.getVirtualFileRef("/mainFile.cpp", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(SourceFile, std::move(Buf));

  FileEntryRef HeaderFile = FileMgr.getVirtualFileRef(
      "/test-header.h", HeaderBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(HeaderFile, std::move(HeaderBuf));

  FileID MainFileID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
  FileID HeaderFileID = SourceMgr.getOrCreateFileID(HeaderFile, SrcMgr::C_User);
  SourceMgr.setMainFileID(MainFileID);

  auto BeginLoc = SourceMgr.getLocForStartOfFile(MainFileID);
  auto EndLoc = SourceMgr.getLocForEndOfFile(MainFileID);

  auto BeginEOLLoc = SourceMgr.translateLineCol(MainFileID, 1, 7);

  auto HeaderLoc = SourceMgr.getLocForStartOfFile(HeaderFileID);

  EXPECT_EQ(BeginLoc.printToString(SourceMgr), "/mainFile.cpp:1:1");
  EXPECT_EQ(EndLoc.printToString(SourceMgr), "/mainFile.cpp:4:7");

  EXPECT_EQ(BeginEOLLoc.printToString(SourceMgr), "/mainFile.cpp:1:7");
  EXPECT_EQ(HeaderLoc.printToString(SourceMgr), "/test-header.h:1:1");

  EXPECT_EQ(SourceRange(BeginLoc, BeginLoc).printToString(SourceMgr),
            "</mainFile.cpp:1:1>");
  EXPECT_EQ(SourceRange(BeginLoc, BeginEOLLoc).printToString(SourceMgr),
            "</mainFile.cpp:1:1, col:7>");
  EXPECT_EQ(SourceRange(BeginLoc, EndLoc).printToString(SourceMgr),
            "</mainFile.cpp:1:1, line:4:7>");
  EXPECT_EQ(SourceRange(BeginLoc, HeaderLoc).printToString(SourceMgr),
            "</mainFile.cpp:1:1, /test-header.h:1:1>");
}

TEST_F(SourceManagerTest, getInvalidBOM) {
  ASSERT_EQ(SrcMgr::ContentCache::getInvalidBOM(""), nullptr);
  ASSERT_EQ(SrcMgr::ContentCache::getInvalidBOM("\x00\x00\x00"), nullptr);
  ASSERT_EQ(SrcMgr::ContentCache::getInvalidBOM("\xFF\xFF\xFF"), nullptr);
  ASSERT_EQ(SrcMgr::ContentCache::getInvalidBOM("#include <iostream>"),
            nullptr);

  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\xFE\xFF#include <iostream>")),
            "UTF-16 (BE)");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\xFF\xFE#include <iostream>")),
            "UTF-16 (LE)");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\x2B\x2F\x76#include <iostream>")),
            "UTF-7");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\xF7\x64\x4C#include <iostream>")),
            "UTF-1");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\xDD\x73\x66\x73#include <iostream>")),
            "UTF-EBCDIC");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\x0E\xFE\xFF#include <iostream>")),
            "SCSU");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\xFB\xEE\x28#include <iostream>")),
            "BOCU-1");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                "\x84\x31\x95\x33#include <iostream>")),
            "GB-18030");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                llvm::StringLiteral::withInnerNUL(
                    "\x00\x00\xFE\xFF#include <iostream>"))),
            "UTF-32 (BE)");
  ASSERT_EQ(StringRef(SrcMgr::ContentCache::getInvalidBOM(
                llvm::StringLiteral::withInnerNUL(
                    "\xFF\xFE\x00\x00#include <iostream>"))),
            "UTF-32 (LE)");
}

// Regression test - there was an out of bound access for buffers not terminated by zero.
TEST_F(SourceManagerTest, getLineNumber) {
  const unsigned pageSize = llvm::sys::Process::getPageSizeEstimate();
  std::unique_ptr<char[]> source(new char[pageSize]);
  for(unsigned i = 0; i < pageSize; ++i) {
    source[i] = 'a';
  }

  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(
        llvm::MemoryBufferRef(
          llvm::StringRef(source.get(), 3), "whatever"
        ),
        false
      );

  FileID mainFileID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(mainFileID);

  ASSERT_NO_FATAL_FAILURE(SourceMgr.getLineNumber(mainFileID, 1, nullptr));
}

struct FakeExternalSLocEntrySource : ExternalSLocEntrySource {
  bool ReadSLocEntry(int ID) override { return {}; }
  int getSLocEntryID(SourceLocation::UIntTy SLocOffset) override { return 0; }
  std::pair<SourceLocation, StringRef> getModuleImportLoc(int ID) override {
    return {};
  }
};

TEST_F(SourceManagerTest, loadedSLocEntryIsInTheSameTranslationUnit) {
  auto InSameTU = [=](int LID, int RID) {
    return SourceMgr.isInTheSameTranslationUnitImpl(
        std::make_pair(SourceManagerTestHelper::makeFileID(LID), 0),
        std::make_pair(SourceManagerTestHelper::makeFileID(RID), 0));
  };

  FakeExternalSLocEntrySource ExternalSource;
  SourceMgr.setExternalSLocEntrySource(&ExternalSource);

  unsigned ANumFileIDs = 10;
  auto [AFirstID, X] = SourceMgr.AllocateLoadedSLocEntries(ANumFileIDs, 10);
  int ALastID = AFirstID + ANumFileIDs - 1;
  // FileID(-11)..FileID(-2)
  ASSERT_EQ(AFirstID, -11);
  ASSERT_EQ(ALastID, -2);

  unsigned BNumFileIDs = 20;
  auto [BFirstID, Y] = SourceMgr.AllocateLoadedSLocEntries(BNumFileIDs, 20);
  int BLastID = BFirstID + BNumFileIDs - 1;
  // FileID(-31)..FileID(-12)
  ASSERT_EQ(BFirstID, -31);
  ASSERT_EQ(BLastID, -12);

  // Loaded vs local.
  EXPECT_FALSE(InSameTU(-2, 1));

  // Loaded in the same allocation A.
  EXPECT_TRUE(InSameTU(-11, -2));
  EXPECT_TRUE(InSameTU(-11, -6));

  // Loaded in the same allocation B.
  EXPECT_TRUE(InSameTU(-31, -12));
  EXPECT_TRUE(InSameTU(-31, -16));

  // Loaded from different allocations A and B.
  EXPECT_FALSE(InSameTU(-12, -11));
}

#if defined(LLVM_ON_UNIX)

// A single SourceManager instance is sometimes reused across multiple
// compilations. This test makes sure we're resetting caches built for tracking
// include locations that are based on FileIDs, to make sure we don't report
// wrong include locations when FileIDs coincide between two different runs.
TEST_F(SourceManagerTest, ResetsIncludeLocMap) {
  auto ParseFile = [&] {
    TrivialModuleLoader ModLoader;
    HeaderSearchOptions HSOpts;
    PreprocessorOptions PPOpts;
    HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, &*Target);
    Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup=*/nullptr, /*OwnsHeaderSearch=*/false);
    PP.Initialize(*Target);
    PP.EnterMainSourceFile();
    PP.LexTokensUntilEOF();
    EXPECT_FALSE(Diags.hasErrorOccurred());
  };

  auto Buf = llvm::MemoryBuffer::getMemBuffer("");
  FileEntryRef HeaderFile =
      FileMgr.getVirtualFileRef("/foo.h", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(HeaderFile, std::move(Buf));

  Buf = llvm::MemoryBuffer::getMemBuffer(R"cpp(#include "/foo.h")cpp");
  FileEntryRef BarFile =
      FileMgr.getVirtualFileRef("/bar.h", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(BarFile, std::move(Buf));
  SourceMgr.createFileID(BarFile, {}, clang::SrcMgr::C_User);

  Buf = llvm::MemoryBuffer::getMemBuffer(R"cpp(#include "/foo.h")cpp");
  FileID MFID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(MFID);

  ParseFile();
  auto FooFID = SourceMgr.getOrCreateFileID(HeaderFile, clang::SrcMgr::C_User);
  auto IncFID = SourceMgr.getDecomposedIncludedLoc(FooFID).first;
  EXPECT_EQ(IncFID, MFID);

  // Clean up source-manager state before we start next parse.
  SourceMgr.clearIDTables();

  // Set up a new main file.
  Buf = llvm::MemoryBuffer::getMemBuffer(R"cpp(
  // silly comment 42
  #include "/bar.h")cpp");
  MFID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(MFID);

  ParseFile();
  // Make sure foo.h got the same file-id in both runs.
  EXPECT_EQ(FooFID,
            SourceMgr.getOrCreateFileID(HeaderFile, clang::SrcMgr::C_User));
  auto BarFID = SourceMgr.getOrCreateFileID(BarFile, clang::SrcMgr::C_User);
  IncFID = SourceMgr.getDecomposedIncludedLoc(FooFID).first;
  // Check that includer is bar.h during this run.
  EXPECT_EQ(IncFID, BarFID);
}

TEST_F(SourceManagerTest, getMacroArgExpandedLocation) {
  const char *header =
    "#define FM(x,y) x\n";

  const char *main =
    "#include \"/test-header.h\"\n"
    "#define VAL 0\n"
    "FM(VAL,0)\n"
    "FM(0,VAL)\n"
    "FM(FM(0,VAL),0)\n"
    "#define CONCAT(X, Y) X##Y\n"
    "CONCAT(1,1)\n";

  std::unique_ptr<llvm::MemoryBuffer> HeaderBuf =
      llvm::MemoryBuffer::getMemBuffer(header);
  std::unique_ptr<llvm::MemoryBuffer> MainBuf =
      llvm::MemoryBuffer::getMemBuffer(main);
  FileID mainFileID = SourceMgr.createFileID(std::move(MainBuf));
  SourceMgr.setMainFileID(mainFileID);

  FileEntryRef headerFile = FileMgr.getVirtualFileRef(
      "/test-header.h", HeaderBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(headerFile, std::move(HeaderBuf));

  HeaderSearchOptions HSOpts;
  PreprocessorOptions PPOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, &*Target);

  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup=*/nullptr, /*OwnsHeaderSearch=*/false);
  // Ensure we can get expanded locations in presence of implicit includes.
  // These are different than normal includes since predefines buffer doesn't
  // have a valid insertion location.
  PP.setPredefines("#include \"/implicit-header.h\"");
  FileMgr.getVirtualFileRef("/implicit-header.h", 0, 0);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  PP.LexTokensUntilEOF(&toks);

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(4U, toks.size());
  ASSERT_EQ(tok::numeric_constant, toks[0].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[1].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[2].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[3].getKind());

  SourceLocation defLoc = SourceMgr.translateLineCol(mainFileID, 2, 13);
  SourceLocation loc1 = SourceMgr.translateLineCol(mainFileID, 3, 8);
  SourceLocation loc2 = SourceMgr.translateLineCol(mainFileID, 4, 4);
  SourceLocation loc3 = SourceMgr.translateLineCol(mainFileID, 5, 7);
  SourceLocation defLoc2 = SourceMgr.translateLineCol(mainFileID, 6, 22);
  defLoc = SourceMgr.getMacroArgExpandedLocation(defLoc);
  loc1 = SourceMgr.getMacroArgExpandedLocation(loc1);
  loc2 = SourceMgr.getMacroArgExpandedLocation(loc2);
  loc3 = SourceMgr.getMacroArgExpandedLocation(loc3);
  defLoc2 = SourceMgr.getMacroArgExpandedLocation(defLoc2);

  EXPECT_TRUE(defLoc.isFileID());
  EXPECT_TRUE(loc1.isFileID());
  EXPECT_TRUE(SourceMgr.isMacroArgExpansion(loc2));
  EXPECT_TRUE(SourceMgr.isMacroArgExpansion(loc3));
  EXPECT_EQ(loc2, toks[1].getLocation());
  EXPECT_EQ(loc3, toks[2].getLocation());
  EXPECT_TRUE(defLoc2.isFileID());
}

namespace {

struct MacroAction {
  enum Kind { kExpansion, kDefinition, kUnDefinition};

  SourceLocation Loc;
  std::string Name;
  LLVM_PREFERRED_TYPE(Kind)
  unsigned MAKind : 3;

  MacroAction(SourceLocation Loc, StringRef Name, unsigned K)
      : Loc(Loc), Name(std::string(Name)), MAKind(K) {}

  bool isExpansion() const { return MAKind == kExpansion; }
  bool isDefinition() const { return MAKind & kDefinition; }
  bool isUnDefinition() const { return MAKind & kUnDefinition; }
};

class MacroTracker : public PPCallbacks {
  std::vector<MacroAction> &Macros;

public:
  explicit MacroTracker(std::vector<MacroAction> &Macros) : Macros(Macros) { }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    Macros.push_back(MacroAction(MD->getLocation(),
                                 MacroNameTok.getIdentifierInfo()->getName(),
                                 MacroAction::kDefinition));
  }
  void MacroUndefined(const Token &MacroNameTok,
                      const MacroDefinition &MD,
                      const MacroDirective  *UD) override {
    Macros.push_back(
        MacroAction(UD ? UD->getLocation() : SourceLocation(),
                    MacroNameTok.getIdentifierInfo()->getName(),
                    UD ? MacroAction::kDefinition | MacroAction::kUnDefinition
                       : MacroAction::kUnDefinition));
  }
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    Macros.push_back(MacroAction(MacroNameTok.getLocation(),
                                 MacroNameTok.getIdentifierInfo()->getName(),
                                 MacroAction::kExpansion));
  }
};

}

TEST_F(SourceManagerTest, isBeforeInTranslationUnitWithMacroInInclude) {
  const char *header =
    "#define MACRO_IN_INCLUDE 0\n"
    "#define MACRO_DEFINED\n"
    "#undef MACRO_DEFINED\n"
    "#undef MACRO_UNDEFINED\n";

  const char *main =
    "#define M(x) x\n"
    "#define INC \"/test-header.h\"\n"
    "#include M(INC)\n"
    "#define INC2 </test-header.h>\n"
    "#include M(INC2)\n";

  std::unique_ptr<llvm::MemoryBuffer> HeaderBuf =
      llvm::MemoryBuffer::getMemBuffer(header);
  std::unique_ptr<llvm::MemoryBuffer> MainBuf =
      llvm::MemoryBuffer::getMemBuffer(main);
  SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(MainBuf)));

  FileEntryRef headerFile = FileMgr.getVirtualFileRef(
      "/test-header.h", HeaderBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(headerFile, std::move(HeaderBuf));

  HeaderSearchOptions HSOpts;
  PreprocessorOptions PPOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, &*Target);
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup=*/nullptr, /*OwnsHeaderSearch=*/false);
  PP.Initialize(*Target);

  std::vector<MacroAction> Macros;
  PP.addPPCallbacks(std::make_unique<MacroTracker>(Macros));

  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  PP.LexTokensUntilEOF(&toks);

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(0U, toks.size());

  ASSERT_EQ(15U, Macros.size());
  // #define M(x) x
  ASSERT_TRUE(Macros[0].isDefinition());
  ASSERT_EQ("M", Macros[0].Name);
  // #define INC "/test-header.h"
  ASSERT_TRUE(Macros[1].isDefinition());
  ASSERT_EQ("INC", Macros[1].Name);
  // M expansion in #include M(INC)
  ASSERT_FALSE(Macros[2].isDefinition());
  ASSERT_EQ("M", Macros[2].Name);
  // INC expansion in #include M(INC)
  ASSERT_TRUE(Macros[3].isExpansion());
  ASSERT_EQ("INC", Macros[3].Name);
  // #define MACRO_IN_INCLUDE 0
  ASSERT_TRUE(Macros[4].isDefinition());
  ASSERT_EQ("MACRO_IN_INCLUDE", Macros[4].Name);
  // #define MACRO_DEFINED
  ASSERT_TRUE(Macros[5].isDefinition());
  ASSERT_FALSE(Macros[5].isUnDefinition());
  ASSERT_EQ("MACRO_DEFINED", Macros[5].Name);
  // #undef MACRO_DEFINED
  ASSERT_TRUE(Macros[6].isDefinition());
  ASSERT_TRUE(Macros[6].isUnDefinition());
  ASSERT_EQ("MACRO_DEFINED", Macros[6].Name);
  // #undef MACRO_UNDEFINED
  ASSERT_FALSE(Macros[7].isDefinition());
  ASSERT_TRUE(Macros[7].isUnDefinition());
  ASSERT_EQ("MACRO_UNDEFINED", Macros[7].Name);
  // #define INC2 </test-header.h>
  ASSERT_TRUE(Macros[8].isDefinition());
  ASSERT_EQ("INC2", Macros[8].Name);
  // M expansion in #include M(INC2)
  ASSERT_FALSE(Macros[9].isDefinition());
  ASSERT_EQ("M", Macros[9].Name);
  // INC2 expansion in #include M(INC2)
  ASSERT_TRUE(Macros[10].isExpansion());
  ASSERT_EQ("INC2", Macros[10].Name);
  // #define MACRO_IN_INCLUDE 0
  ASSERT_TRUE(Macros[11].isDefinition());
  ASSERT_EQ("MACRO_IN_INCLUDE", Macros[11].Name);

  // The INC expansion in #include M(INC) comes before the first
  // MACRO_IN_INCLUDE definition of the included file.
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(Macros[3].Loc, Macros[4].Loc));

  // The INC2 expansion in #include M(INC2) comes before the second
  // MACRO_IN_INCLUDE definition of the included file.
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(Macros[10].Loc, Macros[11].Loc));
}

TEST_F(SourceManagerTest, isMainFile) {
  const char *Source = "int x;";

  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileEntryRef SourceFile =
      FileMgr.getVirtualFileRef("mainFile.cpp", Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(SourceFile, std::move(Buf));

  std::unique_ptr<llvm::MemoryBuffer> Buf2 =
      llvm::MemoryBuffer::getMemBuffer(Source);
  FileEntryRef SecondFile =
      FileMgr.getVirtualFileRef("file2.cpp", Buf2->getBufferSize(), 0);
  SourceMgr.overrideFileContents(SecondFile, std::move(Buf2));

  FileID MainFileID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
  SourceMgr.setMainFileID(MainFileID);

  EXPECT_TRUE(SourceMgr.isMainFile(*SourceFile));
  EXPECT_TRUE(SourceMgr.isMainFile(*SourceFile));
  EXPECT_FALSE(SourceMgr.isMainFile(*SecondFile));
}

#endif

} // anonymous namespace
