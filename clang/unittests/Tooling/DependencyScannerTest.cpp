//===- DependencyScannerTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/DependencyScanningTool.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Prints out all of the gathered dependencies into a string.
class TestFileCollector : public DependencyFileGenerator {
public:
  TestFileCollector(DependencyOutputOptions &Opts,
                    std::vector<std::string> &Deps)
      : DependencyFileGenerator(Opts), Deps(Deps) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    auto NewDeps = getDependencies();
    llvm::append_range(Deps, NewDeps);
  }

private:
  std::vector<std::string> &Deps;
};

// FIXME: Use the regular Service/Worker/Collector APIs instead of
//        reimplementing the action.
class TestDependencyScanningAction : public tooling::ToolAction {
public:
  TestDependencyScanningAction(std::vector<std::string> &Deps) : Deps(Deps) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    CompilerInstance Compiler(std::move(Invocation),
                              std::move(PCHContainerOps));
    Compiler.setVirtualFileSystem(FileMgr->getVirtualFileSystemPtr());
    Compiler.setFileManager(FileMgr);
    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    Compiler.createSourceManager();
    Compiler.addDependencyCollector(std::make_shared<TestFileCollector>(
        Compiler.getInvocation().getDependencyOutputOpts(), Deps));

    auto Action = std::make_unique<PreprocessOnlyAction>();
    return Compiler.ExecuteAction(*Action);
  }

private:
  std::vector<std::string> &Deps;
};

} // namespace

TEST(DependencyScanner, ScanDepsReuseFilemanager) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"symlink.h\"\n#include \"header.h\"\n"));

  ClangTool Tool(CDB, {"test.cpp"}, std::make_shared<PCHContainerOperations>(),
                 VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  // The first invocation should return dependencies in order of access.
  ASSERT_EQ(Deps.size(), 3u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/header.h");

  // The file manager should still have two FileEntries, as one file is a
  // hardlink.
  FileManager &Files = Tool.getFiles();
  EXPECT_EQ(Files.getNumUniqueRealFiles(), 2u);

  Deps.clear();
  Tool.run(&Action);
  // The second invocation should have the same order of dependencies.
  ASSERT_EQ(Deps.size(), 3u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/header.h");

  EXPECT_EQ(Files.getNumUniqueRealFiles(), 2u);
}

TEST(DependencyScanner, ScanDepsReuseFilemanagerSkippedFile) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));
  std::string Test2Path =
      std::string(llvm::formatv("{0}root{0}test2.cpp", Sept));

  VFS->addFile(HeaderPath, 0,
               llvm::MemoryBuffer::getMemBuffer("#pragma once\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"header.h\"\n#include \"symlink.h\"\n"));
  VFS->addFile(Test2Path, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"symlink.h\"\n#include \"header.h\"\n"));

  ClangTool Tool(CDB, {"test.cpp", "test2.cpp"},
                 std::make_shared<PCHContainerOperations>(), VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  ASSERT_EQ(Deps.size(), 6u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[3]), "/root/test2.cpp");
  EXPECT_EQ(convert_to_slash(Deps[4]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[5]), "/root/header.h");
}

TEST(DependencyScanner, ScanDepsReuseFilemanagerHasInclude) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(
      TestPath, 0,
      llvm::MemoryBuffer::getMemBuffer("#if __has_include(\"header.h\") && "
                                       "__has_include(\"symlink.h\")\n#endif"));

  ClangTool Tool(CDB, {"test.cpp", "test.cpp"},
                 std::make_shared<PCHContainerOperations>(), VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  ASSERT_EQ(Deps.size(), 6u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[3]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[4]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[5]), "/root/symlink.h");
}

TEST(DependencyScanner, ScanDepsWithFS) {
  std::vector<std::string> CommandLine = {"clang",
                                          "-target",
                                          "x86_64-apple-macosx10.7",
                                          "-c",
                                          "test.cpp",
                                          "-o"
                                          "test.cpp.o"};
  StringRef CWD = "/root";

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer("#include \"header.h\"\n"));

  DependencyScanningServiceOptions Opts;
  Opts.MakeVFS = [&] { return VFS; };
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningTool ScanTool(Service);

  TextDiagnosticBuffer DiagConsumer;
  std::optional<std::string> DepFile = ScanTool.getDependencyFile(
      CommandLine, CWD, CallbackActionController::lookupUnreachableModuleOutput,
      DiagConsumer);
  ASSERT_TRUE(DepFile.has_value());
  EXPECT_EQ(llvm::sys::path::convert_to_slash(*DepFile),
            "test.cpp.o: /root/test.cpp /root/header.h\n");
}

TEST(DependencyScanner, ScanDepsWithModuleLookup) {
  std::vector<std::string> CommandLine = {
      "clang",
      "-target",
      "x86_64-apple-macosx10.7",
      "-c",
      "test.m",
      "-o"
      "test.m.o",
      "-fmodules",
      "-I/root/SomeSources",
  };
  StringRef CWD = "/root";

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string OtherPath =
      std::string(llvm::formatv("{0}root{0}SomeSources{0}other.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.m", Sept));

  VFS->addFile(OtherPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addFile(TestPath, 0, llvm::MemoryBuffer::getMemBuffer("@import Foo;\n"));

  struct InterceptorFS : llvm::vfs::ProxyFileSystem {
    std::vector<std::string> StatPaths;
    std::vector<std::string> ReadFiles;

    InterceptorFS(IntrusiveRefCntPtr<FileSystem> UnderlyingFS)
        : ProxyFileSystem(UnderlyingFS) {}

    llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
      StatPaths.push_back(Path.str());
      return ProxyFileSystem::status(Path);
    }

    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
    openFileForRead(const Twine &Path) override {
      ReadFiles.push_back(Path.str());
      return ProxyFileSystem::openFileForRead(Path);
    }
  };

  auto InterceptFS = llvm::makeIntrusiveRefCnt<InterceptorFS>(VFS);

  DependencyScanningServiceOptions Opts;
  Opts.MakeVFS = [&] { return InterceptFS; };
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningTool ScanTool(Service);

  // This will fail with "fatal error: module 'Foo' not found" but it doesn't
  // matter, the point of the test is to check that files are not read
  // unnecessarily.
  TextDiagnosticBuffer DiagConsumer;
  std::optional<std::string> DepFile = ScanTool.getDependencyFile(
      CommandLine, CWD, CallbackActionController::lookupUnreachableModuleOutput,
      DiagConsumer);
  ASSERT_FALSE(DepFile.has_value());

  EXPECT_TRUE(!llvm::is_contained(InterceptFS->StatPaths, OtherPath));
  EXPECT_EQ(InterceptFS->ReadFiles, std::vector<std::string>{"test.m"});
}

// When scanning from a TU buffer, the in-memory TU lives ONLY
// in the overlay filesystem built from that buffer, never on the base VFS. If
// DependencyScanningWorker::computeDependencies moves the overlay away before
// initializing the scan CompilerInstance, the scanner falls back to the base
// VFS, cannot find its input, and the scan fails.
TEST(DependencyScanner, ScanDepsTUBufferOverlayReachesScan) {
  std::vector<std::string> CommandLine = {
      "clang", "-target", "x86_64-apple-macosx10.7", "-c", "-o", "tu.o"};
  StringRef CWD = "/root";

  // Base VFS intentionally does NOT contain the TU file.
  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);

  DependencyScanningServiceOptions Opts;
  Opts.MakeVFS = [&] { return VFS; };
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningTool ScanTool(Service);

  auto Sept = llvm::sys::path::get_separator();
  std::string TUPath = std::string(llvm::formatv("{0}root{0}tu.c", Sept));
  auto TU = llvm::MemoryBuffer::getMemBuffer("int main(void) { return 0; }\n",
                                             TUPath);

  TextDiagnosticBuffer DiagConsumer;
  llvm::DenseSet<ModuleID> AlreadySeen;
  auto Result = ScanTool.getTranslationUnitDependencies(
      CommandLine, CWD, DiagConsumer, AlreadySeen,
      CallbackActionController::lookupUnreachableModuleOutput,
      TU->getMemBufferRef());
  ASSERT_TRUE(Result.has_value());
  EXPECT_TRUE(llvm::any_of(Result->FileDeps,
                           [](StringRef F) { return F.contains("tu.c"); }));
}

// A service is tied to a single build, so a path invalidated after the service
// built a module depending on it doesn't cause a rebuild. Rebuilding would
// write a different module file than the one already in the service's module
// cache.
TEST(DependencyScanner, InvalidatedPathAfterModuleBuilt) {
  SmallString<128> Root;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("invalidated-path", Root));
  auto Path = [&](StringRef Rel) {
    SmallString<128> P(Root);
    llvm::sys::path::append(P, Rel);
    llvm::sys::path::native(P);
    return std::string(P);
  };
  auto Write = [&](StringRef Rel, StringRef Contents) {
    ASSERT_FALSE(llvm::sys::fs::create_directories(
        llvm::sys::path::parent_path(Path(Rel))));
    std::error_code EC;
    llvm::raw_fd_ostream OS(Path(Rel), EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  };
  Write("include/module.modulemap", "module Umb { umbrella \"dir\" }\n");
  Write("include/dir/a.h", "");
  Write("tu.c", "#include \"dir/a.h\"\n");

  std::vector<std::string> CommandLine = {"clang",
                                          "-fmodules",
                                          "-fmodules-cache-path=" +
                                              Path("cache"),
                                          "-I" + Path("include"),
                                          "-c",
                                          "tu.c",
                                          "-o",
                                          "tu.o"};

  DependencyScanningServiceOptions Opts;
  Opts.ValidateAgainstInvalidatedPaths = true;
  DependencyScanningService Service(std::move(Opts));
  DependencyScanningTool ScanTool(Service);
  auto ScanModuleFileDeps = [&]() -> std::vector<std::string> {
    TextDiagnosticBuffer DiagConsumer;
    auto Result = ScanTool.getTranslationUnitDependencies(
        CommandLine, Root, DiagConsumer, {},
        [&](const ModuleDeps &MD, ModuleOutputKind) {
          return Path("out/" + MD.ID.ModuleName + ".pcm");
        });
    if (!Result || Result->ModuleGraph.size() != 1)
      return {};
    std::vector<std::string> Deps;
    Result->ModuleGraph[0].forEachFileDep([&](StringRef File) {
      Deps.push_back(llvm::sys::path::filename(File).str());
    });
    return Deps;
  };

  std::vector<std::string> Expected = {"module.modulemap", "a.h"};
  EXPECT_EQ(ScanModuleFileDeps(), Expected);

  Write("include/dir/b.h", "");
  Service.addInvalidatedPath(Path("include/dir"));
  EXPECT_EQ(ScanModuleFileDeps(), Expected);

  llvm::sys::fs::remove_directories(Root);
}

/// Test fixture for module map resolution without module loading.
class GetModuleMapsForModulesTest : public ::testing::Test {
protected:
  GetModuleMapsForModulesTest()
      : VFS(llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>()),
        Service(makeServiceOptions(VFS)), ScanTool(Service) {
    VFS->setCurrentWorkingDirectory(CWD);
  }

  void addFile(StringRef Path, StringRef Contents) {
    VFS->addFile(Path, 0, llvm::MemoryBuffer::getMemBufferCopy(Contents));
  }

  /// Wrapper to get the defined modulemaps for /p ModuleNames, without loading.
  std::optional<llvm::StringMap<std::string>>
  getModuleMaps(ArrayRef<std::string> CommandLine,
                ArrayRef<StringRef> ModuleNames,
                DiagnosticConsumer &DiagConsumer) {
    CallbackActionController Controller(
        CallbackActionController::lookupUnreachableModuleOutput);
    return ScanTool.getModuleMapsForModules(CWD, CommandLine, DiagConsumer,
                                            Controller, ModuleNames);
  }

  /// Get the modulemap that defines the module \p Name by doing a full module
  /// load. If not found, return an empty string.
  std::string scanModuleMapFile(ArrayRef<std::string> CommandLine,
                                StringRef Name,
                                DiagnosticConsumer &DiagConsumer) {
    llvm::DenseSet<ModuleID> AlreadySeen;
    FullDependencyConsumer Consumer(AlreadySeen);
    auto LookupModuleOutput = [](const ModuleDeps &MD, ModuleOutputKind) {
      return "/root/out/" + MD.ID.ModuleName + ".pcm";
    };
    CallbackActionController Controller(LookupModuleOutput);
    bool Queried = false;
    auto GetNextName = [&Queried, Name]() -> std::optional<std::string> {
      if (Queried)
        return std::nullopt;
      Queried = true;
      return Name.str();
    };
    if (!ScanTool.getByNameDependencies(CWD, CommandLine, DiagConsumer,
                                        Controller, GetNextName, Consumer))
      return "";
    TranslationUnitDeps Deps = Consumer.takeTranslationUnitDeps();
    for (const ModuleDeps &MD : Deps.ModuleGraph)
      if (MD.ID.ModuleName == Name)
        return MD.ClangModuleMapFile;
    return "";
  }

  static constexpr StringRef CWD = "/root";
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  DependencyScanningService Service;
  DependencyScanningTool ScanTool;

private:
  static DependencyScanningServiceOptions
  makeServiceOptions(IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS) {
    DependencyScanningServiceOptions Opts;
    Opts.MakeVFS = [VFS] { return VFS; };
    return Opts;
  }
};

TEST_F(GetModuleMapsForModulesTest, Found) {
  std::vector<std::string> CommandLine = {
      "clang",     "-target",  "x86_64-apple-macosx10.7",
      "-c",        "test.m",   "-fmodules",
      "-I/root/a", "-I/root/b"};
  addFile("/root/test.m", "\n");
  addFile("/root/a/module.modulemap", "module A { header \"a.h\" }\n");
  addFile("/root/a/a.h", "\n");
  addFile("/root/b/module.modulemap", "module B { header \"b.h\" }\n");
  addFile("/root/b/b.h", "\n");

  TextDiagnosticBuffer DiagConsumer;
  std::optional<llvm::StringMap<std::string>> ModuleMaps =
      getModuleMaps(CommandLine, {"A", "B"}, DiagConsumer);
  ASSERT_TRUE(ModuleMaps);
  using llvm::sys::path::convert_to_slash;
  EXPECT_EQ(ModuleMaps->size(), 2u);
  EXPECT_EQ(convert_to_slash(ModuleMaps->lookup("A")),
            "/root/a/module.modulemap");
  EXPECT_EQ(convert_to_slash(ModuleMaps->lookup("B")),
            "/root/b/module.modulemap");
}

// Verify a module provided by a prebuilt module file is omitted, even though
// it's resolvable through search paths.
TEST_F(GetModuleMapsForModulesTest, PrebuiltModule) {
  // Set up a scenario where 'A' is a prebuilt module but its modulemap is
  // resolved during the lookup for a different module 'B'.
  std::vector<std::string> CommandLine = {
      "clang",  "-target",   "x86_64-apple-macosx10.7",     "-c",
      "test.m", "-fmodules", "-fmodule-file=A=/root/a.pcm", "-I/root/a"};
  addFile("/root/test.m", "\n");
  addFile("/root/a.pcm", "");
  addFile("/root/a/module.modulemap", "module A { header \"a.h\" }\n"
                                      "module B { header \"b.h\" }\n");
  addFile("/root/a/a.h", "\n");
  addFile("/root/a/b.h", "\n");

  TextDiagnosticBuffer DiagConsumer;
  std::optional<llvm::StringMap<std::string>> ModuleMaps =
      getModuleMaps(CommandLine, {"A", "B"}, DiagConsumer);
  ASSERT_TRUE(ModuleMaps);
  EXPECT_EQ(ModuleMaps->size(), 1u);
  EXPECT_EQ(llvm::sys::path::convert_to_slash(ModuleMaps->lookup("B")),
            "/root/a/module.modulemap");
  EXPECT_FALSE(ModuleMaps->contains("A"));
}

// Verify a module that is not found is omitted, leaving an empty result.
TEST_F(GetModuleMapsForModulesTest, NotFound) {
  std::vector<std::string> CommandLine = {
      "clang",     "-target",  "x86_64-apple-macosx10.7", "-c", "test.m",
      "-fmodules", "-I/root/a"};
  addFile("/root/test.m", "\n");
  addFile("/root/a/a.h", "\n");

  TextDiagnosticBuffer DiagConsumer;
  std::optional<llvm::StringMap<std::string>> ModuleMaps =
      getModuleMaps(CommandLine, {"Missing"}, DiagConsumer);
  ASSERT_TRUE(ModuleMaps);
  EXPECT_TRUE(ModuleMaps->empty());
}

// Verify a command line that cannot be set up reports
// an error.
TEST_F(GetModuleMapsForModulesTest, IllFormedCommandLine) {
  std::vector<std::string> CommandLine = {"clang", "-cc1", "-not-a-real-flag"};

  TextDiagnosticBuffer DiagConsumer;
  std::optional<llvm::StringMap<std::string>> ModuleMaps =
      getModuleMaps(CommandLine, {"A"}, DiagConsumer);
  EXPECT_FALSE(ModuleMaps);
  ASSERT_NE(DiagConsumer.err_begin(), DiagConsumer.err_end());
  EXPECT_EQ(DiagConsumer.err_begin()->second,
            "unknown argument: '-not-a-real-flag'");
}

// Verify a resolution error isn't diagnosed by the lookup, but is by a
// later scan.
TEST_F(GetModuleMapsForModulesTest, SkipsRedefinition) {
  // Set up module redefinition error scenario.
  // Both search paths define a module named 'A', and the compiler
  // instance encounters both.
  std::vector<std::string> CommandLine = {
      "clang",     "-target",  "x86_64-apple-macosx10.7",
      "-c",        "test.m",   "-fmodules",
      "-I/root/a", "-I/root/b"};
  addFile("/root/test.m", "\n");
  addFile("/root/a/module.modulemap", "module A { header \"a.h\" }\n");
  addFile("/root/a/a.h", "\n");
  addFile("/root/b/module.modulemap", "module A { header \"a.h\" }\n"
                                      "module B { header \"b.h\" }\n");
  addFile("/root/b/a.h", "\n");
  addFile("/root/b/b.h", "\n");

  auto getErrors = [](TextDiagnosticBuffer &DiagConsumer) {
    std::vector<std::string> Errors;
    for (auto It = DiagConsumer.err_begin(); It != DiagConsumer.err_end(); ++It)
      Errors.push_back(It->second);
    return Errors;
  };

  TextDiagnosticBuffer LookupDiags;
  std::optional<llvm::StringMap<std::string>> ModuleMaps =
      getModuleMaps(CommandLine, {"A", "B"}, LookupDiags);
  ASSERT_TRUE(ModuleMaps);
  using llvm::sys::path::convert_to_slash;
  EXPECT_EQ(convert_to_slash(ModuleMaps->lookup("A")),
            "/root/a/module.modulemap");
  EXPECT_EQ(convert_to_slash(ModuleMaps->lookup("B")),
            "/root/b/module.modulemap");
  EXPECT_EQ(getErrors(LookupDiags), std::vector<std::string>{});

  TextDiagnosticBuffer ScanDiags;
  EXPECT_EQ(scanModuleMapFile(CommandLine, "B", ScanDiags), "");
  EXPECT_TRUE(
      llvm::is_contained(getErrors(ScanDiags), "redefinition of module 'A'"));
}
