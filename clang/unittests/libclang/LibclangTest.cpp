//===- unittests/libclang/LibclangTest.cpp --- libclang tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
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

class LibclangParseTest : public ::testing::Test {
  std::set<std::string> Files;
  typedef std::unique_ptr<std::string> fixed_addr_string;
  std::map<fixed_addr_string, fixed_addr_string> UnsavedFileContents;
public:
  std::string TestDir;
  CXIndex Index;
  CXTranslationUnit ClangTU;
  unsigned TUFlags;
  std::vector<CXUnsavedFile> UnsavedFiles;

  void SetUp() override {
    llvm::SmallString<256> Dir;
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("libclang-test", Dir));
    TestDir = Dir.str();
    TUFlags = CXTranslationUnit_DetailedPreprocessingRecord |
      clang_defaultEditingTranslationUnitOptions();
    Index = clang_createIndex(0, 0);
    ClangTU = nullptr;
  }
  void TearDown() override {
    clang_disposeTranslationUnit(ClangTU);
    clang_disposeIndex(Index);
    for (const std::string &Path : Files)
      llvm::sys::fs::remove(Path);
    llvm::sys::fs::remove(TestDir);
  }
  void WriteFile(std::string &Filename, const std::string &Contents) {
    if (!llvm::sys::path::is_absolute(Filename)) {
      llvm::SmallString<256> Path(TestDir);
      llvm::sys::path::append(Path, Filename);
      Filename = Path.str();
      Files.insert(Filename);
    }
    llvm::sys::fs::create_directories(llvm::sys::path::parent_path(Filename));
    std::ofstream OS(Filename);
    OS << Contents;
    assert(OS.good());
  }
  void MapUnsavedFile(std::string Filename, const std::string &Contents) {
    if (!llvm::sys::path::is_absolute(Filename)) {
      llvm::SmallString<256> Path(TestDir);
      llvm::sys::path::append(Path, Filename);
      Filename = Path.str();
    }
    auto it = UnsavedFileContents.insert(std::make_pair(
        fixed_addr_string(new std::string(Filename)),
        fixed_addr_string(new std::string(Contents))));
    UnsavedFiles.push_back({
        it.first->first->c_str(),   // filename
        it.first->second->c_str(),  // contents
        it.first->second->size()    // length
    });
  }
  template<typename F>
  void Traverse(const F &TraversalFunctor) {
    CXCursor TuCursor = clang_getTranslationUnitCursor(ClangTU);
    std::reference_wrapper<const F> FunctorRef = std::cref(TraversalFunctor);
    clang_visitChildren(TuCursor,
        &TraverseStateless<std::reference_wrapper<const F>>,
        &FunctorRef);
  }
private:
  template<typename TState>
  static CXChildVisitResult TraverseStateless(CXCursor cx, CXCursor parent,
      CXClientData data) {
    TState *State = static_cast<TState*>(data);
    return State->get()(cx, parent);
  }
};

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

class LibclangReparseTest : public LibclangParseTest {
public:
  void DisplayDiagnostics() {
    unsigned NumDiagnostics = clang_getNumDiagnostics(ClangTU);
    for (unsigned i = 0; i < NumDiagnostics; ++i) {
      auto Diag = clang_getDiagnostic(ClangTU, i);
      DEBUG(llvm::dbgs() << clang_getCString(clang_formatDiagnostic(
          Diag, clang_defaultDiagnosticDisplayOptions())) << "\n");
      clang_disposeDiagnostic(Diag);
    }
  }
  bool ReparseTU(unsigned num_unsaved_files, CXUnsavedFile* unsaved_files) {
    if (clang_reparseTranslationUnit(ClangTU, num_unsaved_files, unsaved_files,
                                     clang_defaultReparseOptions(ClangTU))) {
      DEBUG(llvm::dbgs() << "Reparse failed\n");
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
  ASSERT_TRUE(strstr(clang_getCString(cxname), CppName.c_str()));
  clang_disposeString(cxname);

  cxname = clang_File_tryGetRealPathName(cxf);
  ASSERT_TRUE(strstr(clang_getCString(cxname), CppName.c_str()));
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
      DEBUG(llvm::dbgs() << "Saving failed\n");
      return false;
    }

    clang_disposeTranslationUnit(ClangTU);

    ClangTU = clang_createTranslationUnit(Index, Filename.c_str());

    if (!ClangTU) {
      DEBUG(llvm::dbgs() << "Loading failed\n");
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
