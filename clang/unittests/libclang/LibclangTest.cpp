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

class LibclangReparseTest : public ::testing::Test {
  std::set<std::string> Files;
public:
  std::string TestDir;
  CXIndex Index;
  CXTranslationUnit ClangTU;
  unsigned TUFlags;

  void SetUp() override {
    llvm::SmallString<256> Dir;
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("libclang-test", Dir));
    TestDir = Dir.str();
    TUFlags = CXTranslationUnit_DetailedPreprocessingRecord |
              clang_defaultEditingTranslationUnitOptions();
    Index = clang_createIndex(0, 0);
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
