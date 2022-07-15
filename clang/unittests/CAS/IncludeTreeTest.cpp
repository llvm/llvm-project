#include "clang/CAS/IncludeTree.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::cas;

using namespace tooling;
using namespace dependencies;

TEST(IncludeTree, IncludeTreeScan) {
  std::shared_ptr<CASDB> DB = llvm::cas::createInMemoryCAS();
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto add = [&](StringRef Path, StringRef Contents) {
    FS->addFile(Path, 0, llvm::MemoryBuffer::getMemBuffer(Contents));
  };
  StringRef MainContents = R"(
    #include "a1.h"
    #include "sys.h"
  )";
  StringRef A1Contents = R"(
    #if __has_include("other.h")
      #include "other.h"
    #endif
    #if __has_include("b1.h")
      #include "b1.h"
    #endif
  )";
  add("t.cpp", MainContents);
  add("a1.h", A1Contents);
  add("b1.h", "");
  add("sys/sys.h", "");
  std::unique_ptr<llvm::vfs::FileSystem> VFS =
      llvm::cas::createCASProvidingFileSystem(DB, FS);

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::IncludeTree,
                                    CASOptions(), nullptr);
  DependencyScanningTool ScanTool(Service, std::move(VFS));

  std::vector<std::string> CommandLine = {"clang",
                                          "-target",
                                          "x86_64-apple-macos11",
                                          "-isystem",
                                          "sys",
                                          "-c",
                                          "t.cpp",
                                          "-o"
                                          "t.cpp.o"};
  Optional<IncludeTreeRoot> Root;
  ASSERT_THAT_ERROR(
      ScanTool.getIncludeTree(*DB, CommandLine, /*CWD*/ "").moveInto(Root),
      llvm::Succeeded());
  Optional<IncludeTree> Main;
  ASSERT_THAT_ERROR(Root->getMainFileTree().moveInto(Main), llvm::Succeeded());
  {
    EXPECT_EQ(Main->getFileCharacteristic(), SrcMgr::C_User);
    IncludeFile::FileInfo FI;
    ASSERT_THAT_ERROR(Main->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "t.cpp");
    EXPECT_EQ(FI.Contents, MainContents);
  }
  ASSERT_EQ(Main->getNumIncludes(), uint32_t(3));

  Optional<IncludeTree> Predef;
  ASSERT_THAT_ERROR(Main->getInclude(0).moveInto(Predef), llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(0), uint32_t(0));
  {
    EXPECT_EQ(Predef->getFileCharacteristic(), SrcMgr::C_User);
    IncludeFile::FileInfo FI;
    ASSERT_THAT_ERROR(Predef->getBaseFileInfo().moveInto(FI),
                      llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "<built-in>");
  }

  Optional<IncludeTree> A1;
  ASSERT_THAT_ERROR(Main->getInclude(1).moveInto(A1), llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(1), uint32_t(21));
  {
    EXPECT_EQ(A1->getFileCharacteristic(), SrcMgr::C_User);
    IncludeFile::FileInfo FI;
    ASSERT_THAT_ERROR(A1->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "./a1.h");
    EXPECT_EQ(FI.Contents, A1Contents);
    EXPECT_FALSE(A1->getCheckResult(0));
    EXPECT_TRUE(A1->getCheckResult(1));

    ASSERT_EQ(A1->getNumIncludes(), uint32_t(1));
    Optional<IncludeTree> B1;
    ASSERT_THAT_ERROR(A1->getInclude(0).moveInto(B1), llvm::Succeeded());
    EXPECT_EQ(A1->getIncludeOffset(0), uint32_t(122));
    {
      EXPECT_EQ(B1->getFileCharacteristic(), SrcMgr::C_User);
      IncludeFile::FileInfo FI;
      ASSERT_THAT_ERROR(B1->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
      EXPECT_EQ(FI.Filename, "./b1.h");
      EXPECT_EQ(FI.Contents, "");

      ASSERT_EQ(B1->getNumIncludes(), uint32_t(0));
    }
  }

  Optional<IncludeTree> Sys;
  ASSERT_THAT_ERROR(Main->getInclude(2).moveInto(Sys), llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(2), uint32_t(42));
  {
    EXPECT_EQ(Sys->getFileCharacteristic(), SrcMgr::C_System);
    IncludeFile::FileInfo FI;
    ASSERT_THAT_ERROR(Sys->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "sys/sys.h");
    EXPECT_EQ(FI.Contents, "");

    ASSERT_EQ(Sys->getNumIncludes(), uint32_t(0));
  }
}
