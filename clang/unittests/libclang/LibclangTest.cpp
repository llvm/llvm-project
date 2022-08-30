//===- unittests/libclang/LibclangTest.cpp --- libclang tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "clang-c/Rewrite.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include "TestUtils.h"
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#define DEBUG_TYPE "libclang-test"

TEST(libclang, clang_parseTranslationUnit2_InvalidArgs) {
  EXPECT_EQ(CXError_InvalidArguments,
            clang_parseTranslationUnit2(nullptr, nullptr, nullptr, 0, nullptr,
                                        0, 0, nullptr));
}

TEST(libclang, clang_createTranslationUnit_InvalidArgs) {
  EXPECT_EQ(nullptr, clang_createTranslationUnit(nullptr, nullptr));
}

TEST(libclang, clang_createTranslationUnit2_InvalidArgs) {
  EXPECT_EQ(CXError_InvalidArguments,
            clang_createTranslationUnit2(nullptr, nullptr, nullptr));

  CXTranslationUnit TU = reinterpret_cast<CXTranslationUnit>(1);
  EXPECT_EQ(CXError_InvalidArguments,
            clang_createTranslationUnit2(nullptr, nullptr, &TU));
  EXPECT_EQ(nullptr, TU);
}

namespace {
struct TestVFO {
  const char *Contents;
  CXVirtualFileOverlay VFO;

  TestVFO(const char *Contents) : Contents(Contents) {
    VFO = clang_VirtualFileOverlay_create(0);
  }

  void map(const char *VPath, const char *RPath) {
    CXErrorCode Err = clang_VirtualFileOverlay_addFileMapping(VFO, VPath, RPath);
    EXPECT_EQ(Err, CXError_Success);
  }

  void mapError(const char *VPath, const char *RPath, CXErrorCode ExpErr) {
    CXErrorCode Err = clang_VirtualFileOverlay_addFileMapping(VFO, VPath, RPath);
    EXPECT_EQ(Err, ExpErr);
  }

  ~TestVFO() {
    if (Contents) {
      char *BufPtr;
      unsigned BufSize;
      clang_VirtualFileOverlay_writeToBuffer(VFO, 0, &BufPtr, &BufSize);
      std::string BufStr(BufPtr, BufSize);
      EXPECT_STREQ(Contents, BufStr.c_str());
      clang_free(BufPtr);
    }
    clang_VirtualFileOverlay_dispose(VFO);
  }
};
}

TEST(libclang, VirtualFileOverlay_Basic) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/virtual\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo.h\",\n"
      "          'external-contents': \"/real/foo.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/virtual/foo.h", "/real/foo.h");
}

TEST(libclang, VirtualFileOverlay_Unicode) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/\\u266B\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"\\u2602.h\",\n"
      "          'external-contents': \"/real/\\u2602.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/♫/☂.h", "/real/☂.h");
}

TEST(libclang, VirtualFileOverlay_InvalidArgs) {
  TestVFO T(nullptr);
  T.mapError("/path/./virtual/../foo.h", "/real/foo.h",
             CXError_InvalidArguments);
}

TEST(libclang, VirtualFileOverlay_RemapDirectories) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/another/dir\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo2.h\",\n"
      "          'external-contents': \"/real/foo2.h\"\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/virtual/dir\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo1.h\",\n"
      "          'external-contents': \"/real/foo1.h\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo3.h\",\n"
      "          'external-contents': \"/real/foo3.h\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'directory',\n"
      "          'name': \"in/subdir\",\n"
      "          'contents': [\n"
      "            {\n"
      "              'type': 'file',\n"
      "              'name': \"foo4.h\",\n"
      "              'external-contents': \"/real/foo4.h\"\n"
      "            }\n"
      "          ]\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/virtual/dir/foo1.h", "/real/foo1.h");
  T.map("/another/dir/foo2.h", "/real/foo2.h");
  T.map("/path/virtual/dir/foo3.h", "/real/foo3.h");
  T.map("/path/virtual/dir/in/subdir/foo4.h", "/real/foo4.h");
}

TEST(libclang, VirtualFileOverlay_CaseInsensitive) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'case-sensitive': 'false',\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/virtual\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo.h\",\n"
      "          'external-contents': \"/real/foo.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/virtual/foo.h", "/real/foo.h");
  clang_VirtualFileOverlay_setCaseSensitivity(T.VFO, false);
}

TEST(libclang, VirtualFileOverlay_SharedPrefix) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/foo\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"bar\",\n"
      "          'external-contents': \"/real/bar\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"bar.h\",\n"
      "          'external-contents': \"/real/bar.h\"\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/foobar\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"baz.h\",\n"
      "          'external-contents': \"/real/baz.h\"\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foobarbaz.h\",\n"
      "          'external-contents': \"/real/foobarbaz.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/foo/bar.h", "/real/bar.h");
  T.map("/path/foo/bar", "/real/bar");
  T.map("/path/foobar/baz.h", "/real/baz.h");
  T.map("/path/foobarbaz.h", "/real/foobarbaz.h");
}

TEST(libclang, VirtualFileOverlay_AdjacentDirectory) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/dir1\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo.h\",\n"
      "          'external-contents': \"/real/foo.h\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'directory',\n"
      "          'name': \"subdir\",\n"
      "          'contents': [\n"
      "            {\n"
      "              'type': 'file',\n"
      "              'name': \"bar.h\",\n"
      "              'external-contents': \"/real/bar.h\"\n"
      "            }\n"
      "          ]\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/dir2\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"baz.h\",\n"
      "          'external-contents': \"/real/baz.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/dir1/foo.h", "/real/foo.h");
  T.map("/path/dir1/subdir/bar.h", "/real/bar.h");
  T.map("/path/dir2/baz.h", "/real/baz.h");
}

TEST(libclang, VirtualFileOverlay_TopLevel) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo.h\",\n"
      "          'external-contents': \"/real/foo.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/foo.h", "/real/foo.h");
}

TEST(libclang, VirtualFileOverlay_Empty) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
}

TEST(libclang, ModuleMapDescriptor) {
  const char *Contents =
    "framework module TestFrame {\n"
    "  umbrella header \"TestFrame.h\"\n"
    "\n"
    "  export *\n"
    "  module * { export * }\n"
    "}\n";

  CXModuleMapDescriptor MMD = clang_ModuleMapDescriptor_create(0);

  clang_ModuleMapDescriptor_setFrameworkModuleName(MMD, "TestFrame");
  clang_ModuleMapDescriptor_setUmbrellaHeader(MMD, "TestFrame.h");

  char *BufPtr;
  unsigned BufSize;
  clang_ModuleMapDescriptor_writeToBuffer(MMD, 0, &BufPtr, &BufSize);
  std::string BufStr(BufPtr, BufSize);
  EXPECT_STREQ(Contents, BufStr.c_str());
  clang_free(BufPtr);
  clang_ModuleMapDescriptor_dispose(MMD);
}

TEST_F(LibclangParseTest, AllSkippedRanges) {
  std::string Header = "header.h", Main = "main.cpp";
  WriteFile(Header,
    "#ifdef MANGOS\n"
    "printf(\"mmm\");\n"
    "#endif");
  WriteFile(Main,
    "#include \"header.h\"\n"
    "#ifdef KIWIS\n"
    "printf(\"mmm!!\");\n"
    "#endif");

  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0,
                                       nullptr, 0, TUFlags);

  CXSourceRangeList *Ranges = clang_getAllSkippedRanges(ClangTU);
  EXPECT_EQ(2U, Ranges->count);
  
  CXSourceLocation cxl;
  unsigned line;
  cxl = clang_getRangeStart(Ranges->ranges[0]);
  clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
  EXPECT_EQ(1U, line);
  cxl = clang_getRangeEnd(Ranges->ranges[0]);
  clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
  EXPECT_EQ(3U, line);

  cxl = clang_getRangeStart(Ranges->ranges[1]);
  clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
  EXPECT_EQ(2U, line);
  cxl = clang_getRangeEnd(Ranges->ranges[1]);
  clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
  EXPECT_EQ(4U, line);

  clang_disposeSourceRangeList(Ranges);
}

TEST_F(LibclangParseTest, EvaluateChildExpression) {
  std::string Main = "main.m";
  WriteFile(Main, "#define kFOO @\"foo\"\n"
                  "void foobar(void) {\n"
                  " {kFOO;}\n"
                  "}\n");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_FunctionDecl) {
          int numberedStmt = 0;
          clang_visitChildren(
              cursor,
              [](CXCursor cursor, CXCursor parent,
                 CXClientData client_data) -> CXChildVisitResult {
                int &numberedStmt = *((int *)client_data);
                if (clang_getCursorKind(cursor) == CXCursor_CompoundStmt) {
                  if (numberedStmt) {
                    CXEvalResult RE = clang_Cursor_Evaluate(cursor);
                    EXPECT_NE(RE, nullptr);
                    EXPECT_EQ(clang_EvalResult_getKind(RE),
                              CXEval_ObjCStrLiteral);
                    clang_EvalResult_dispose(RE);
                    return CXChildVisit_Break;
                  }
                  numberedStmt++;
                }
                return CXChildVisit_Recurse;
              },
              &numberedStmt);
          EXPECT_EQ(numberedStmt, 1);
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

class LibclangReparseTest : public LibclangParseTest {
public:
  void DisplayDiagnostics() {
    unsigned NumDiagnostics = clang_getNumDiagnostics(ClangTU);
    for (unsigned i = 0; i < NumDiagnostics; ++i) {
      auto Diag = clang_getDiagnostic(ClangTU, i);
      LLVM_DEBUG(llvm::dbgs()
                 << clang_getCString(clang_formatDiagnostic(
                        Diag, clang_defaultDiagnosticDisplayOptions()))
                 << "\n");
      clang_disposeDiagnostic(Diag);
    }
  }
  bool ReparseTU(unsigned num_unsaved_files, CXUnsavedFile* unsaved_files) {
    if (clang_reparseTranslationUnit(ClangTU, num_unsaved_files, unsaved_files,
                                     clang_defaultReparseOptions(ClangTU))) {
      LLVM_DEBUG(llvm::dbgs() << "Reparse failed\n");
      return false;
    }
    DisplayDiagnostics();
    return true;
  }
};

TEST_F(LibclangReparseTest, FileName) {
  std::string CppName = "main.cpp";
  WriteFile(CppName, "int main() {}");
  ClangTU = clang_parseTranslationUnit(Index, CppName.c_str(), nullptr, 0,
                                       nullptr, 0, TUFlags);
  CXFile cxf = clang_getFile(ClangTU, CppName.c_str());

  CXString cxname = clang_getFileName(cxf);
  ASSERT_STREQ(clang_getCString(cxname), CppName.c_str());
  clang_disposeString(cxname);

  cxname = clang_File_tryGetRealPathName(cxf);
  ASSERT_TRUE(llvm::StringRef(clang_getCString(cxname)).endswith("main.cpp"));
  clang_disposeString(cxname);
}

TEST_F(LibclangReparseTest, Reparse) {
  const char *HeaderTop = "#ifndef H\n#define H\nstruct Foo { int bar;";
  const char *HeaderBottom = "\n};\n#endif\n";
  const char *CppFile = "#include \"HeaderFile.h\"\nint main() {"
                         " Foo foo; foo.bar = 7; foo.baz = 8; }\n";
  std::string HeaderName = "HeaderFile.h";
  std::string CppName = "CppFile.cpp";
  WriteFile(CppName, CppFile);
  WriteFile(HeaderName, std::string(HeaderTop) + HeaderBottom);

  ClangTU = clang_parseTranslationUnit(Index, CppName.c_str(), nullptr, 0,
                                       nullptr, 0, TUFlags);
  EXPECT_EQ(1U, clang_getNumDiagnostics(ClangTU));
  DisplayDiagnostics();

  // Immedaitely reparse.
  ASSERT_TRUE(ReparseTU(0, nullptr /* No unsaved files. */));
  EXPECT_EQ(1U, clang_getNumDiagnostics(ClangTU));

  std::string NewHeaderContents =
      std::string(HeaderTop) + "int baz;" + HeaderBottom;
  WriteFile(HeaderName, NewHeaderContents);

  // Reparse after fix.
  ASSERT_TRUE(ReparseTU(0, nullptr /* No unsaved files. */));
  EXPECT_EQ(0U, clang_getNumDiagnostics(ClangTU));
}

TEST_F(LibclangReparseTest, ReparseWithModule) {
  const char *HeaderTop = "#ifndef H\n#define H\nstruct Foo { int bar;";
  const char *HeaderBottom = "\n};\n#endif\n";
  const char *MFile = "#include \"HeaderFile.h\"\nint main() {"
                         " struct Foo foo; foo.bar = 7; foo.baz = 8; }\n";
  const char *ModFile = "module A { header \"HeaderFile.h\" }\n";
  std::string HeaderName = "HeaderFile.h";
  std::string MName = "MFile.m";
  std::string ModName = "module.modulemap";
  WriteFile(MName, MFile);
  WriteFile(HeaderName, std::string(HeaderTop) + HeaderBottom);
  WriteFile(ModName, ModFile);

  std::string ModulesCache = std::string("-fmodules-cache-path=") + TestDir;
  const char *Args[] = { "-fmodules", ModulesCache.c_str(),
                         "-I", TestDir.c_str() };
  int NumArgs = sizeof(Args) / sizeof(Args[0]);
  ClangTU = clang_parseTranslationUnit(Index, MName.c_str(), Args, NumArgs,
                                       nullptr, 0, TUFlags);
  EXPECT_EQ(1U, clang_getNumDiagnostics(ClangTU));
  DisplayDiagnostics();

  // Immedaitely reparse.
  ASSERT_TRUE(ReparseTU(0, nullptr /* No unsaved files. */));
  EXPECT_EQ(1U, clang_getNumDiagnostics(ClangTU));

  std::string NewHeaderContents =
      std::string(HeaderTop) + "int baz;" + HeaderBottom;
  WriteFile(HeaderName, NewHeaderContents);

  // Reparse after fix.
  ASSERT_TRUE(ReparseTU(0, nullptr /* No unsaved files. */));
  EXPECT_EQ(0U, clang_getNumDiagnostics(ClangTU));
}

TEST_F(LibclangReparseTest, clang_parseTranslationUnit2FullArgv) {
  // Provide a fake GCC 99.9.9 standard library that always overrides any local
  // GCC installation.
  std::string EmptyFiles[] = {"lib/gcc/arm-linux-gnueabi/99.9.9/crtbegin.o",
                              "include/arm-linux-gnueabi/.keep",
                              "include/c++/99.9.9/vector"};

  for (auto &Name : EmptyFiles)
    WriteFile(Name, "\n");

  std::string Filename = "test.cc";
  WriteFile(Filename, "#include <vector>\n");

  std::string Clang = "bin/clang";
  WriteFile(Clang, "");

  const char *Argv[] = {Clang.c_str(), "-target", "arm-linux-gnueabi",
                        "-stdlib=libstdc++", "--gcc-toolchain="};

  EXPECT_EQ(CXError_Success,
            clang_parseTranslationUnit2FullArgv(Index, Filename.c_str(), Argv,
                                                sizeof(Argv) / sizeof(Argv[0]),
                                                nullptr, 0, TUFlags, &ClangTU));
  EXPECT_EQ(0U, clang_getNumDiagnostics(ClangTU));
  DisplayDiagnostics();
}

class LibclangPrintingPolicyTest : public LibclangParseTest {
public:
  CXPrintingPolicy Policy = nullptr;

  void SetUp() override {
    LibclangParseTest::SetUp();
    std::string File = "file.cpp";
    WriteFile(File, "int i;\n");
    ClangTU = clang_parseTranslationUnit(Index, File.c_str(), nullptr, 0,
                                         nullptr, 0, TUFlags);
    CXCursor TUCursor = clang_getTranslationUnitCursor(ClangTU);
    Policy = clang_getCursorPrintingPolicy(TUCursor);
  }
  void TearDown() override {
    clang_PrintingPolicy_dispose(Policy);
    LibclangParseTest::TearDown();
  }
};

TEST_F(LibclangPrintingPolicyTest, SetAndGetProperties) {
  for (unsigned Value = 0; Value < 2; ++Value) {
    for (int I = 0; I < CXPrintingPolicy_LastProperty; ++I) {
      auto Property = static_cast<enum CXPrintingPolicyProperty>(I);

      clang_PrintingPolicy_setProperty(Policy, Property, Value);
      EXPECT_EQ(Value, clang_PrintingPolicy_getProperty(Policy, Property));
    }
  }
}

TEST_F(LibclangReparseTest, PreprocessorSkippedRanges) {
  std::string Header = "header.h", Main = "main.cpp";
  WriteFile(Header,
    "#ifdef MANGOS\n"
    "printf(\"mmm\");\n"
    "#endif");
  WriteFile(Main,
    "#include \"header.h\"\n"
    "#ifdef GUAVA\n"
    "#endif\n"
    "#ifdef KIWIS\n"
    "printf(\"mmm!!\");\n"
    "#endif");

  for (int i = 0; i != 3; ++i) {
    unsigned flags = TUFlags | CXTranslationUnit_PrecompiledPreamble;
    if (i == 2)
      flags |= CXTranslationUnit_CreatePreambleOnFirstParse;

    if (i != 0)
       clang_disposeTranslationUnit(ClangTU);  // dispose from previous iter

    // parse once
    ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0,
                                         nullptr, 0, flags);
    if (i != 0) {
      // reparse
      ASSERT_TRUE(ReparseTU(0, nullptr /* No unsaved files. */));
    }

    // Check all ranges are there
    CXSourceRangeList *Ranges = clang_getAllSkippedRanges(ClangTU);
    EXPECT_EQ(3U, Ranges->count);

    CXSourceLocation cxl;
    unsigned line;
    cxl = clang_getRangeStart(Ranges->ranges[0]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(1U, line);
    cxl = clang_getRangeEnd(Ranges->ranges[0]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(3U, line);

    cxl = clang_getRangeStart(Ranges->ranges[1]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(2U, line);
    cxl = clang_getRangeEnd(Ranges->ranges[1]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(3U, line);

    cxl = clang_getRangeStart(Ranges->ranges[2]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(4U, line);
    cxl = clang_getRangeEnd(Ranges->ranges[2]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(6U, line);

    clang_disposeSourceRangeList(Ranges);

    // Check obtaining ranges by each file works
    CXFile cxf = clang_getFile(ClangTU, Header.c_str());
    Ranges = clang_getSkippedRanges(ClangTU, cxf);
    EXPECT_EQ(1U, Ranges->count);
    cxl = clang_getRangeStart(Ranges->ranges[0]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(1U, line);
    clang_disposeSourceRangeList(Ranges);

    cxf = clang_getFile(ClangTU, Main.c_str());
    Ranges = clang_getSkippedRanges(ClangTU, cxf);
    EXPECT_EQ(2U, Ranges->count);
    cxl = clang_getRangeStart(Ranges->ranges[0]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(2U, line);
    cxl = clang_getRangeStart(Ranges->ranges[1]);
    clang_getSpellingLocation(cxl, nullptr, &line, nullptr, nullptr);
    EXPECT_EQ(4U, line);
    clang_disposeSourceRangeList(Ranges);
  }
}

class LibclangSerializationTest : public LibclangParseTest {
public:
  bool SaveAndLoadTU(const std::string &Filename) {
    unsigned options = clang_defaultSaveOptions(ClangTU);
    if (clang_saveTranslationUnit(ClangTU, Filename.c_str(), options) !=
        CXSaveError_None) {
      LLVM_DEBUG(llvm::dbgs() << "Saving failed\n");
      return false;
    }

    clang_disposeTranslationUnit(ClangTU);

    ClangTU = clang_createTranslationUnit(Index, Filename.c_str());

    if (!ClangTU) {
      LLVM_DEBUG(llvm::dbgs() << "Loading failed\n");
      return false;
    }

    return true;
  }
};

TEST_F(LibclangSerializationTest, TokenKindsAreCorrectAfterLoading) {
  // Ensure that "class" is recognized as a keyword token after serializing
  // and reloading the AST, as it is not a keyword for the default LangOptions.
  std::string HeaderName = "test.h";
  WriteFile(HeaderName, "enum class Something {};");

  const char *Argv[] = {"-xc++-header", "-std=c++11"};

  ClangTU = clang_parseTranslationUnit(Index, HeaderName.c_str(), Argv,
                                       sizeof(Argv) / sizeof(Argv[0]), nullptr,
                                       0, TUFlags);

  auto CheckTokenKinds = [=]() {
    CXSourceRange Range =
        clang_getCursorExtent(clang_getTranslationUnitCursor(ClangTU));

    CXToken *Tokens;
    unsigned int NumTokens;
    clang_tokenize(ClangTU, Range, &Tokens, &NumTokens);

    ASSERT_EQ(6u, NumTokens);
    EXPECT_EQ(CXToken_Keyword, clang_getTokenKind(Tokens[0]));
    EXPECT_EQ(CXToken_Keyword, clang_getTokenKind(Tokens[1]));
    EXPECT_EQ(CXToken_Identifier, clang_getTokenKind(Tokens[2]));
    EXPECT_EQ(CXToken_Punctuation, clang_getTokenKind(Tokens[3]));
    EXPECT_EQ(CXToken_Punctuation, clang_getTokenKind(Tokens[4]));
    EXPECT_EQ(CXToken_Punctuation, clang_getTokenKind(Tokens[5]));

    clang_disposeTokens(ClangTU, Tokens, NumTokens);
  };

  CheckTokenKinds();

  std::string ASTName = "test.ast";
  WriteFile(ASTName, "");

  ASSERT_TRUE(SaveAndLoadTU(ASTName));

  CheckTokenKinds();
}

TEST_F(LibclangParseTest, clang_getVarDeclInitializer) {
  std::string Main = "main.cpp";
  WriteFile(Main, "int foo() { return 5; }; const int a = foo();");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_VarDecl) {
          const CXCursor Initializer = clang_Cursor_getVarDeclInitializer(cursor);
          EXPECT_FALSE(clang_Cursor_isNull(Initializer));
          CXString Spelling = clang_getCursorSpelling(Initializer);
          const char* const SpellingCSstr = clang_getCString(Spelling);
          EXPECT_TRUE(SpellingCSstr);
          EXPECT_EQ(std::string(SpellingCSstr), std::string("foo"));
          clang_disposeString(Spelling);
          return CXChildVisit_Break;
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

TEST_F(LibclangParseTest, clang_hasVarDeclGlobalStorageFalse) {
  std::string Main = "main.cpp";
  WriteFile(Main, "void foo() { int a; }");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_VarDecl) {
          EXPECT_FALSE(clang_Cursor_hasVarDeclGlobalStorage(cursor));
          return CXChildVisit_Break;
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

TEST_F(LibclangParseTest, clang_Cursor_hasVarDeclGlobalStorageTrue) {
  std::string Main = "main.cpp";
  WriteFile(Main, "int a;");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_VarDecl) {
          EXPECT_TRUE(clang_Cursor_hasVarDeclGlobalStorage(cursor));
          return CXChildVisit_Break;
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

TEST_F(LibclangParseTest, clang_Cursor_hasVarDeclExternalStorageFalse) {
  std::string Main = "main.cpp";
  WriteFile(Main, "int a;");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_VarDecl) {
          EXPECT_FALSE(clang_Cursor_hasVarDeclExternalStorage(cursor));
          return CXChildVisit_Break;
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

TEST_F(LibclangParseTest, clang_Cursor_hasVarDeclExternalStorageTrue) {
  std::string Main = "main.cpp";
  WriteFile(Main, "extern int a;");
  ClangTU = clang_parseTranslationUnit(Index, Main.c_str(), nullptr, 0, nullptr,
                                       0, TUFlags);

  CXCursor C = clang_getTranslationUnitCursor(ClangTU);
  clang_visitChildren(
      C,
      [](CXCursor cursor, CXCursor parent,
         CXClientData client_data) -> CXChildVisitResult {
        if (clang_getCursorKind(cursor) == CXCursor_VarDecl) {
          EXPECT_TRUE(clang_Cursor_hasVarDeclExternalStorage(cursor));
          return CXChildVisit_Break;
        }
        return CXChildVisit_Continue;
      },
      nullptr);
}

TEST_F(LibclangParseTest, clang_getUnqualifiedTypeRemovesQualifiers) {
  std::string Header = "header.h";
  WriteFile(Header, "void foo1(const int);\n"
                    "void foo2(volatile int);\n"
                    "void foo3(const volatile int);\n"
                    "void foo4(int* const);\n"
                    "void foo5(int* volatile);\n"
                    "void foo6(int* restrict);\n"
                    "void foo7(int* const volatile);\n"
                    "void foo8(int* volatile restrict);\n"
                    "void foo9(int* const restrict);\n"
                    "void foo10(int* const volatile restrict);\n");

  auto is_qualified = [](CXType type) -> bool {
    return clang_isConstQualifiedType(type) ||
           clang_isVolatileQualifiedType(type) ||
           clang_isRestrictQualifiedType(type);
  };

  auto from_CXString = [](CXString cx_string) -> std::string {
    std::string string{clang_getCString(cx_string)};

    clang_disposeString(cx_string);

    return string;
  };

  ClangTU = clang_parseTranslationUnit(Index, Header.c_str(), nullptr, 0,
                                       nullptr, 0, TUFlags);

  Traverse([&is_qualified, &from_CXString](CXCursor cursor, CXCursor) {
    if (clang_getCursorKind(cursor) == CXCursor_FunctionDecl) {
      CXType arg_type = clang_getArgType(clang_getCursorType(cursor), 0);
      EXPECT_TRUE(is_qualified(arg_type))
          << "Input data '" << from_CXString(clang_getCursorSpelling(cursor))
          << "' first argument does not have a qualified type.";

      CXType unqualified_arg_type = clang_getUnqualifiedType(arg_type);
      EXPECT_FALSE(is_qualified(unqualified_arg_type))
          << "The type '" << from_CXString(clang_getTypeSpelling(arg_type))
          << "' was not unqualified after a call to clang_getUnqualifiedType.";
    }

    return CXChildVisit_Continue;
  });
}

class LibclangRewriteTest : public LibclangParseTest {
public:
  CXRewriter Rew = nullptr;
  std::string Filename;
  CXFile File = nullptr;

  void SetUp() override {
    LibclangParseTest::SetUp();
    Filename = "file.cpp";
    WriteFile(Filename, "int main() { return 0; }");
    ClangTU = clang_parseTranslationUnit(Index, Filename.c_str(), nullptr, 0,
                                         nullptr, 0, TUFlags);
    Rew = clang_CXRewriter_create(ClangTU);
    File = clang_getFile(ClangTU, Filename.c_str());
  }
  void TearDown() override {
    clang_CXRewriter_dispose(Rew);
    LibclangParseTest::TearDown();
  }
};

static std::string getFileContent(const std::string& Filename) {
  std::ifstream RewrittenFile(Filename);
  std::string RewrittenFileContent;
  std::string Line;
  while (std::getline(RewrittenFile, Line)) {
    if (RewrittenFileContent.empty())
      RewrittenFileContent = Line;
    else {
      RewrittenFileContent += "\n" + Line;
    }
  }
  return RewrittenFileContent;
}

TEST_F(LibclangRewriteTest, RewriteReplace) {
  CXSourceLocation B = clang_getLocation(ClangTU, File, 1, 5);
  CXSourceLocation E = clang_getLocation(ClangTU, File, 1, 9);
  CXSourceRange Rng	= clang_getRange(B, E);

  clang_CXRewriter_replaceText(Rew, Rng, "MAIN");

  ASSERT_EQ(clang_CXRewriter_overwriteChangedFiles(Rew), 0);
  EXPECT_EQ(getFileContent(Filename), "int MAIN() { return 0; }");
}

TEST_F(LibclangRewriteTest, RewriteReplaceShorter) {
  CXSourceLocation B = clang_getLocation(ClangTU, File, 1, 5);
  CXSourceLocation E = clang_getLocation(ClangTU, File, 1, 9);
  CXSourceRange Rng	= clang_getRange(B, E);

  clang_CXRewriter_replaceText(Rew, Rng, "foo");

  ASSERT_EQ(clang_CXRewriter_overwriteChangedFiles(Rew), 0);
  EXPECT_EQ(getFileContent(Filename), "int foo() { return 0; }");
}

TEST_F(LibclangRewriteTest, RewriteReplaceLonger) {
  CXSourceLocation B = clang_getLocation(ClangTU, File, 1, 5);
  CXSourceLocation E = clang_getLocation(ClangTU, File, 1, 9);
  CXSourceRange Rng	= clang_getRange(B, E);

  clang_CXRewriter_replaceText(Rew, Rng, "patatino");

  ASSERT_EQ(clang_CXRewriter_overwriteChangedFiles(Rew), 0);
  EXPECT_EQ(getFileContent(Filename), "int patatino() { return 0; }");
}

TEST_F(LibclangRewriteTest, RewriteInsert) {
  CXSourceLocation Loc = clang_getLocation(ClangTU, File, 1, 5);

  clang_CXRewriter_insertTextBefore(Rew, Loc, "ro");

  ASSERT_EQ(clang_CXRewriter_overwriteChangedFiles(Rew), 0);
  EXPECT_EQ(getFileContent(Filename), "int romain() { return 0; }");
}

TEST_F(LibclangRewriteTest, RewriteRemove) {
  CXSourceLocation B = clang_getLocation(ClangTU, File, 1, 5);
  CXSourceLocation E = clang_getLocation(ClangTU, File, 1, 9);
  CXSourceRange Rng	= clang_getRange(B, E);

  clang_CXRewriter_removeText(Rew, Rng);

  ASSERT_EQ(clang_CXRewriter_overwriteChangedFiles(Rew), 0);
  EXPECT_EQ(getFileContent(Filename), "int () { return 0; }");
}
