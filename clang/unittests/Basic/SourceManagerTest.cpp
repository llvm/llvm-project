//===- unittests/Basic/SourceManagerTest.cpp ------ SourceManager tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "gtest/gtest.h"

using namespace clang;

namespace {

// The test fixture.
class SourceManagerTest : public ::testing::Test {
protected:
  SourceManagerTest()
    : FileMgr(FileMgrOpts),
      DiagID(new DiagnosticIDs()),
      Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr),
      TargetOpts(new TargetOptions) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

class VoidModuleLoader : public ModuleLoader {
  ModuleLoadResult loadModule(SourceLocation ImportLoc, 
                              ModuleIdPath Path,
                              Module::NameVisibilityKind Visibility,
                              bool IsInclusionDirective) override {
    return ModuleLoadResult();
  }

  void makeModuleVisible(Module *Mod,
                         Module::NameVisibilityKind Visibility,
                         SourceLocation ImportLoc) override { }

  GlobalModuleIndex *loadGlobalModuleIndex(SourceLocation TriggerLoc) override
    { return nullptr; }
  bool lookupMissingImports(StringRef Name, SourceLocation TriggerLoc) override
    { return 0; }
};

TEST_F(SourceManagerTest, isBeforeInTranslationUnit) {
  const char *source =
    "#define M(x) [x]\n"
    "M(foo)";
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(source);
  FileID mainFileID = SourceMgr.createFileID(std::move(Buf));
  SourceMgr.setMainFileID(mainFileID);

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(new HeaderSearchOptions, SourceMgr, Diags, LangOpts, 
                          &*Target);
  Preprocessor PP(new PreprocessorOptions(), Diags, LangOpts, SourceMgr,
                  HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

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

#if defined(LLVM_ON_UNIX)

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

  const FileEntry *headerFile = FileMgr.getVirtualFile("/test-header.h",
                                                 HeaderBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(headerFile, std::move(HeaderBuf));

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(new HeaderSearchOptions, SourceMgr, Diags, LangOpts, 
                          &*Target);
  Preprocessor PP(new PreprocessorOptions(), Diags, LangOpts, SourceMgr,
                  HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

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
  SourceLocation Loc;
  std::string Name;
  bool isDefinition; // if false, it is expansion.
  
  MacroAction(SourceLocation Loc, StringRef Name, bool isDefinition)
    : Loc(Loc), Name(Name), isDefinition(isDefinition) { }
};

class MacroTracker : public PPCallbacks {
  std::vector<MacroAction> &Macros;

public:
  explicit MacroTracker(std::vector<MacroAction> &Macros) : Macros(Macros) { }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    Macros.push_back(MacroAction(MD->getLocation(),
                                 MacroNameTok.getIdentifierInfo()->getName(),
                                 true));
  }
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    Macros.push_back(MacroAction(MacroNameTok.getLocation(),
                                 MacroNameTok.getIdentifierInfo()->getName(),
                                 false));
  }
};

}

TEST_F(SourceManagerTest, isBeforeInTranslationUnitWithMacroInInclude) {
  const char *header =
    "#define MACRO_IN_INCLUDE 0\n";

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

  const FileEntry *headerFile = FileMgr.getVirtualFile("/test-header.h",
                                                 HeaderBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(headerFile, std::move(HeaderBuf));

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(new HeaderSearchOptions, SourceMgr, Diags, LangOpts, 
                          &*Target);
  Preprocessor PP(new PreprocessorOptions(), Diags, LangOpts, SourceMgr,
                  HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);

  std::vector<MacroAction> Macros;
  PP.addPPCallbacks(llvm::make_unique<MacroTracker>(Macros));

  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(0U, toks.size());

  ASSERT_EQ(9U, Macros.size());
  // #define M(x) x
  ASSERT_TRUE(Macros[0].isDefinition);
  ASSERT_EQ("M", Macros[0].Name);
  // #define INC "/test-header.h"
  ASSERT_TRUE(Macros[1].isDefinition);
  ASSERT_EQ("INC", Macros[1].Name);
  // M expansion in #include M(INC)
  ASSERT_FALSE(Macros[2].isDefinition);
  ASSERT_EQ("M", Macros[2].Name);
  // INC expansion in #include M(INC)
  ASSERT_FALSE(Macros[3].isDefinition);
  ASSERT_EQ("INC", Macros[3].Name);
  // #define MACRO_IN_INCLUDE 0
  ASSERT_TRUE(Macros[4].isDefinition);
  ASSERT_EQ("MACRO_IN_INCLUDE", Macros[4].Name);
  // #define INC2 </test-header.h>
  ASSERT_TRUE(Macros[5].isDefinition);
  ASSERT_EQ("INC2", Macros[5].Name);
  // M expansion in #include M(INC2)
  ASSERT_FALSE(Macros[6].isDefinition);
  ASSERT_EQ("M", Macros[6].Name);
  // INC2 expansion in #include M(INC2)
  ASSERT_FALSE(Macros[7].isDefinition);
  ASSERT_EQ("INC2", Macros[7].Name);
  // #define MACRO_IN_INCLUDE 0
  ASSERT_TRUE(Macros[8].isDefinition);
  ASSERT_EQ("MACRO_IN_INCLUDE", Macros[8].Name);

  // The INC expansion in #include M(INC) comes before the first
  // MACRO_IN_INCLUDE definition of the included file.
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(Macros[3].Loc, Macros[4].Loc));

  // The INC2 expansion in #include M(INC2) comes before the second
  // MACRO_IN_INCLUDE definition of the included file.
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(Macros[7].Loc, Macros[8].Loc));
}

#endif

} // anonymous namespace
