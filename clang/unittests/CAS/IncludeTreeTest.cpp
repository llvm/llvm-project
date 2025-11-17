#include "clang/CAS/IncludeTree.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <system_error>

using namespace clang;
using namespace clang::cas;

using namespace tooling;
using namespace dependencies;

TEST(IncludeTree, IncludeTreeScan) {
  StringRef PathSep = llvm::sys::path::get_separator();
  std::shared_ptr<ObjectStore> DB = llvm::cas::createInMemoryCAS();
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->setCurrentWorkingDirectory("/root");
  auto add = [&](StringRef Path, StringRef Contents) {
    FS->addFile(Path, 0, llvm::MemoryBuffer::getMemBuffer(Contents));
  };
  StringRef MainContents = R"(
    #include "a1.h"
    #include "sys.h"
    #include "sys_directive.h"
  )";
  StringRef A1Contents = R"(
    #if __has_include("other.h")
      #include "other.h"
    #endif
    #if __has_include("b1.h")
      #include "b1.h"
    #endif
  )";
  StringRef SysDirectiveContents = R"(
    #pragma clang system_header
    #include "sys_indirect.h"
  )";
  add("t.cpp", MainContents);
  add("a1.h", A1Contents);
  add("b1.h", "");
  add("sys/sys.h", "");
  add("sys_directive.h", SysDirectiveContents);
  add("sys_indirect.h", "");
  std::unique_ptr<llvm::vfs::FileSystem> VFS =
      llvm::cas::createCASProvidingFileSystem(DB, FS);

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::IncludeTree,
                                    CASOptions(), nullptr, nullptr);
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
  std::optional<IncludeTreeRoot> Root;
  ASSERT_THAT_ERROR(
      ScanTool.getIncludeTree(*DB, CommandLine, /*CWD*/ "", nullptr)
          .moveInto(Root),
      llvm::Succeeded());

  std::optional<IncludeTree::File> MainFile;
  std::optional<IncludeTree::File> A1File;
  std::optional<IncludeTree::File> B1File;
  std::optional<IncludeTree::File> SysFile;
  std::optional<IncludeTree::File> SysDirectiveFile;
  std::optional<IncludeTree::File> SysIndirectFile;

  std::optional<IncludeTree> Main;
  ASSERT_THAT_ERROR(Root->getMainFileTree().moveInto(Main), llvm::Succeeded());
  {
    ASSERT_THAT_ERROR(Main->getBaseFile().moveInto(MainFile),
                      llvm::Succeeded());
    EXPECT_EQ(Main->getFileCharacteristic(), SrcMgr::C_User);
    IncludeTree::FileInfo FI;
    ASSERT_THAT_ERROR(MainFile->getFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "t.cpp");
    EXPECT_EQ(FI.Contents, MainContents);
  }
  ASSERT_EQ(Main->getNumIncludes(), uint32_t(4));

  std::optional<IncludeTree> Predef;
  ASSERT_THAT_ERROR(Main->getIncludeTree(0).moveInto(Predef),
                    llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(0), uint32_t(0));
  {
    EXPECT_EQ(Predef->getFileCharacteristic(), SrcMgr::C_User);
    IncludeTree::FileInfo FI;
    ASSERT_THAT_ERROR(Predef->getBaseFileInfo().moveInto(FI),
                      llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "<built-in>");
  }

  std::optional<IncludeTree> A1;
  ASSERT_THAT_ERROR(Main->getIncludeTree(1).moveInto(A1), llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(1), uint32_t(21));
  {
    ASSERT_THAT_ERROR(A1->getBaseFile().moveInto(A1File), llvm::Succeeded());
    EXPECT_EQ(A1->getFileCharacteristic(), SrcMgr::C_User);
    IncludeTree::FileInfo FI;
    ASSERT_THAT_ERROR(A1File->getFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "." + PathSep.str() + "a1.h");
    EXPECT_EQ(FI.Contents, A1Contents);
    EXPECT_FALSE(A1->getCheckResult(0));
    EXPECT_TRUE(A1->getCheckResult(1));

    ASSERT_EQ(A1->getNumIncludes(), uint32_t(1));
    std::optional<IncludeTree> B1;
    ASSERT_THAT_ERROR(A1->getIncludeTree(0).moveInto(B1), llvm::Succeeded());
    EXPECT_EQ(A1->getIncludeOffset(0), uint32_t(122));
    {
      ASSERT_THAT_ERROR(B1->getBaseFile().moveInto(B1File), llvm::Succeeded());
      EXPECT_EQ(B1->getFileCharacteristic(), SrcMgr::C_User);
      IncludeTree::FileInfo FI;
      ASSERT_THAT_ERROR(B1->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
      EXPECT_EQ(FI.Filename, "." + PathSep.str() + "b1.h");
      EXPECT_EQ(FI.Contents, "");

      ASSERT_EQ(B1->getNumIncludes(), uint32_t(0));
    }
  }

  std::optional<IncludeTree> Sys;
  ASSERT_THAT_ERROR(Main->getIncludeTree(2).moveInto(Sys), llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(2), uint32_t(42));
  {
    ASSERT_THAT_ERROR(Sys->getBaseFile().moveInto(SysFile), llvm::Succeeded());
    EXPECT_EQ(Sys->getFileCharacteristic(), SrcMgr::C_System);
    IncludeTree::FileInfo FI;
    ASSERT_THAT_ERROR(Sys->getBaseFileInfo().moveInto(FI), llvm::Succeeded());
    EXPECT_EQ(FI.Filename, "sys" + PathSep.str() + "sys.h");
    EXPECT_EQ(FI.Contents, "");

    ASSERT_EQ(Sys->getNumIncludes(), uint32_t(0));
  }

  std::optional<IncludeTree> SysDirective;
  ASSERT_THAT_ERROR(Main->getIncludeTree(3).moveInto(SysDirective),
                    llvm::Succeeded());
  EXPECT_EQ(Main->getIncludeOffset(3), uint32_t(73));
  {
    ASSERT_THAT_ERROR(SysDirective->getBaseFile().moveInto(SysDirectiveFile),
                      llvm::Succeeded());
    // Note: system_header directive injects a line directive, so C_User is for
    // the start of the file here.
    EXPECT_EQ(SysDirective->getFileCharacteristic(), SrcMgr::C_User);
    ASSERT_EQ(SysDirective->getNumIncludes(), uint32_t(1));
    std::optional<IncludeTree> SysIndirect;
    ASSERT_THAT_ERROR(SysDirective->getIncludeTree(0).moveInto(SysIndirect),
                      llvm::Succeeded());
    {
      ASSERT_THAT_ERROR(SysIndirect->getBaseFile().moveInto(SysIndirectFile),
                        llvm::Succeeded());
      EXPECT_EQ(SysIndirect->getFileCharacteristic(), SrcMgr::C_System);
      ASSERT_EQ(SysIndirect->getNumIncludes(), uint32_t(0));
    }
  }

  std::optional<IncludeTree::FileList> FileList;
  ASSERT_THAT_ERROR(Root->getFileList().moveInto(FileList), llvm::Succeeded());

  SmallVector<std::pair<IncludeTree::File, IncludeTree::FileList::FileSizeTy>>
      Files;
  ASSERT_THAT_ERROR(FileList->forEachFile([&](auto F, auto S) -> llvm::Error {
    Files.push_back({F, S});
    return llvm::Error::success();
  }),
                    llvm::Succeeded());

  ASSERT_EQ(Files.size(), size_t(6));
  EXPECT_EQ(Files[0].first.getRef(), MainFile->getRef());
  EXPECT_EQ(Files[0].second, MainContents.size());
  EXPECT_EQ(Files[1].first.getRef(), A1File->getRef());
  EXPECT_EQ(Files[1].second, A1Contents.size());
  EXPECT_EQ(Files[2].first.getRef(), B1File->getRef());
  EXPECT_EQ(Files[2].second, IncludeTree::FileList::FileSizeTy(0));
  EXPECT_EQ(Files[3].first.getRef(), SysFile->getRef());
  EXPECT_EQ(Files[3].second, IncludeTree::FileList::FileSizeTy(0));
  EXPECT_EQ(Files[4].first.getRef(), SysDirectiveFile->getRef());
  EXPECT_EQ(Files[4].second, SysDirectiveContents.size());
  EXPECT_EQ(Files[5].first.getRef(), SysIndirectFile->getRef());
  EXPECT_EQ(Files[5].second, IncludeTree::FileList::FileSizeTy(0));
}

TEST(IncludeTree, IncludeTreeFileList) {
  std::shared_ptr<ObjectStore> DB = llvm::cas::createInMemoryCAS();
  SmallVector<IncludeTree::File> Files;
  for (unsigned I = 0; I < 10; ++I) {
    std::optional<IncludeTree::File> File;
    std::string Path = "/file" + std::to_string(I);
    static constexpr StringRef Bytes = "123456789";
    std::optional<ObjectRef> Content;
    ASSERT_THAT_ERROR(
        DB->storeFromString({}, Bytes.substr(0, I)).moveInto(Content),
        llvm::Succeeded());
    ASSERT_THAT_ERROR(
        IncludeTree::File::create(*DB, Path, *Content).moveInto(File),
        llvm::Succeeded());
    Files.push_back(std::move(*File));
  }

  auto MakeFileList = [&](unsigned Begin, unsigned End,
                          ArrayRef<ObjectRef> Lists) {
    SmallVector<IncludeTree::FileList::FileEntry> Entries;
    for (; Begin != End; ++Begin)
      Entries.push_back({Files[Begin].getRef(), Begin});
    return IncludeTree::FileList::create(*DB, Entries, Lists);
  };

  std::optional<IncludeTree::FileList> L89, L7, L29, L;
  ASSERT_THAT_ERROR(MakeFileList(8, 10, {}).moveInto(L89), llvm::Succeeded());
  EXPECT_EQ(L89->getNumReferences(), 2u);
  ASSERT_THAT_ERROR(MakeFileList(7, 8, {}).moveInto(L7), llvm::Succeeded());
  EXPECT_EQ(L7->getNumReferences(), 1u);
  ASSERT_THAT_ERROR(
      MakeFileList(2, 7, {L7->getRef(), L89->getRef()}).moveInto(L29),
      llvm::Succeeded());
  EXPECT_EQ(L29->getNumReferences(), 7u); // 2,3,4,5,6, {7}, {8, 9}
  ASSERT_THAT_ERROR(MakeFileList(0, 2, {L29->getRef()}).moveInto(L),
                    llvm::Succeeded());
  EXPECT_EQ(L->getNumReferences(), 3u); // 0, 1, {2, ...}

  size_t I = 0;
  ASSERT_THAT_ERROR(
      L->forEachFile([&](IncludeTree::File F, auto Size) -> llvm::Error {
        if (I >= Files.size())
          return llvm::Error::success(); // diagnosed later.
        EXPECT_EQ(F.getFilenameRef(), Files[I].getFilenameRef())
            << "filename mismatch at " << I;
        EXPECT_EQ(F.getContentsRef(), Files[I].getContentsRef())
            << "contents mismatch at " << I;
        EXPECT_EQ(Size, I) << "size mismatch at " << I;
        I += 1;
        return llvm::Error::success();
      }),
      llvm::Succeeded());
  EXPECT_EQ(I, Files.size());
}

TEST(IncludeTree, IncludeTreeFileListDuplicates) {
  std::shared_ptr<ObjectStore> DB = llvm::cas::createInMemoryCAS();
  SmallVector<IncludeTree::File> Files;
  for (unsigned I = 0; I < 10; ++I) {
    std::optional<IncludeTree::File> File;
    std::string Path = "/file" + std::to_string(I);
    static constexpr StringRef Bytes = "123456789";
    std::optional<ObjectRef> Content;
    ASSERT_THAT_ERROR(
        DB->storeFromString({}, Bytes.substr(0, I)).moveInto(Content),
        llvm::Succeeded());
    ASSERT_THAT_ERROR(
        IncludeTree::File::create(*DB, Path, *Content).moveInto(File),
        llvm::Succeeded());
    Files.push_back(std::move(*File));
  }

  auto MakeFileList = [&](unsigned Begin, unsigned End,
                          ArrayRef<ObjectRef> Lists) {
    SmallVector<IncludeTree::FileList::FileEntry> Entries;
    for (; Begin != End; ++Begin)
      Entries.push_back({Files[Begin].getRef(), Begin});
    return IncludeTree::FileList::create(*DB, Entries, Lists);
  };

  std::optional<IncludeTree::FileList> L89, L;
  ASSERT_THAT_ERROR(MakeFileList(8, 10, {}).moveInto(L89), llvm::Succeeded());
  EXPECT_EQ(L89->getNumReferences(), 2u);
  ASSERT_THAT_ERROR(
      MakeFileList(0, 9, {L89->getRef(), L89->getRef()}).moveInto(L),
      llvm::Succeeded());
  EXPECT_EQ(L->getNumReferences(), 11u); // 0, 1, ..., 8, {8, 9}, {8, 9}

  size_t I = 0;
  ASSERT_THAT_ERROR(
      L->forEachFile([&](IncludeTree::File F, auto Size) -> llvm::Error {
        if (I >= Files.size())
          return llvm::Error::success(); // diagnosed later.
        EXPECT_EQ(F.getFilenameRef(), Files[I].getFilenameRef())
            << "filename mismatch at " << I;
        EXPECT_EQ(F.getContentsRef(), Files[I].getContentsRef())
            << "contents mismatch at " << I;
        EXPECT_EQ(Size, I) << "size mismatch at " << I;
        I += 1;
        return llvm::Error::success();
      }),
      llvm::Succeeded());
  EXPECT_EQ(I, Files.size());
}

TEST(IncludeTree, IncludeTreeFileSystemOverlay) {
  StringRef PathSep = llvm::sys::path::get_separator();
  std::shared_ptr<ObjectStore> DB = llvm::cas::createInMemoryCAS();
  SmallVector<IncludeTree::FileList::FileEntry> Files;
  for (unsigned I = 0; I < 10; ++I) {
    std::optional<IncludeTree::File> File;
    std::string Path = "/file" + std::to_string(I);
    static constexpr StringRef Bytes = "123456789";
    std::optional<ObjectRef> Content;
    ASSERT_THAT_ERROR(
        DB->storeFromString({}, Bytes.substr(0, I)).moveInto(Content),
        llvm::Succeeded());
    ASSERT_THAT_ERROR(
        IncludeTree::File::create(*DB, Path, *Content).moveInto(File),
        llvm::Succeeded());
    Files.push_back({File->getRef(), I});
  }
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> IncludeTreeFS;
  ASSERT_THAT_ERROR(
      createIncludeTreeFileSystem(*DB, Files).moveInto(IncludeTreeFS),
      llvm::Succeeded());

  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->setCurrentWorkingDirectory("/dir");
  FS->addFile("file1", 0,  llvm::MemoryBuffer::getMemBuffer("str"));
  FS->addFile("file2", 0,  llvm::MemoryBuffer::getMemBuffer("other"));

  llvm::vfs::OverlayFileSystem OverlayFS(std::move(FS));
  OverlayFS.pushOverlay(IncludeTreeFS);

  std::error_code EC;
  int NumFile = 0;
  for (auto I = OverlayFS.dir_begin(PathSep.str() + "dir", EC);
       !EC && I != llvm::vfs::directory_iterator(); I.increment(EC)) {
    ASSERT_FALSE(EC);
    ++NumFile;
    std::string Path = PathSep.str() + "dir" + PathSep.str() + "file" +
        std::to_string(NumFile);
    ASSERT_EQ(I->path(), Path);
  }
  ASSERT_EQ(NumFile, 2);
}
