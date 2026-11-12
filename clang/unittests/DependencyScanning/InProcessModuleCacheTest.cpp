#include "clang/DependencyScanning/InProcessModuleCache.h"

#include "clang/Basic/AtomicLineLogger.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace clang::dependencies;

TEST(InProcessModuleCache, ReadWriteInvalidation) {
  ModuleCacheEntries Entries;
  AtomicLineLogger Logger;
  std::shared_ptr<ModuleCache> ModCache =
      makeInProcessModuleCache(Entries, Logger);

  int FD;
  llvm::SmallString<256> Path;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("M", "pcm", FD, Path));

  {
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << "original\n";
  }

  off_t Size;
  time_t ModTime;
  std::unique_ptr<llvm::MemoryBuffer> Buf;
  ASSERT_THAT_ERROR(ModCache->read(Path, Size, ModTime).moveInto(Buf),
                    llvm::Succeeded());
  EXPECT_EQ(Buf->getBuffer(), "original\n");

  auto NewBuf = llvm::MemoryBuffer::getMemBufferCopy("modified\n", Path);
  ASSERT_EQ(ModCache->write(Path, std::move(NewBuf), Size, ModTime).getError(),
            std::error_code{});

  // Writing a new buffer should not invalidate the previously-read buffer.
  EXPECT_EQ(Buf->getBuffer(), "original\n");
}

TEST(InProcessModuleCache, ReadReadInvalidation) {
  ModuleCacheEntries Entries;
  AtomicLineLogger Logger;
  std::shared_ptr<ModuleCache> ModCache =
      makeInProcessModuleCache(Entries, Logger);

  int FD;
  llvm::SmallString<256> Path;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("M", "pcm", FD, Path));

  {
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << "original\n";
  }

  off_t Size1;
  time_t ModTime1;
  std::unique_ptr<llvm::MemoryBuffer> Buf1;
  ASSERT_THAT_ERROR(ModCache->read(Path, Size1, ModTime1).moveInto(Buf1),
                    llvm::Succeeded());
  EXPECT_EQ(Buf1->getBuffer(), "original\n");

  off_t Size2;
  time_t ModTime2;
  std::unique_ptr<llvm::MemoryBuffer> Buf2;
  ASSERT_THAT_ERROR(ModCache->read(Path, Size2, ModTime2).moveInto(Buf2),
                    llvm::Succeeded());
  EXPECT_EQ(Buf2->getBuffer(), "original\n");

  // Subsequent reads should not invalidate previous reads.
  EXPECT_EQ(Buf1->getBuffer(), "original\n");
  // All read buffers should point to the same memory.
  EXPECT_EQ(Buf1->getBuffer().begin(), Buf2->getBuffer().begin());
  EXPECT_EQ(Buf1->getBuffer().end(), Buf2->getBuffer().end());
}

TEST(InProcessModuleCache, WriteReturnsSharedBuffer) {
  ModuleCacheEntries Entries;
  AtomicLineLogger Logger;
  std::shared_ptr<ModuleCache> ModCache =
      makeInProcessModuleCache(Entries, Logger);

  llvm::SmallString<256> Path;
  llvm::sys::fs::createUniquePath("M-%%%%%%.pcm", Path, /*MakeAbsolute=*/true);

  off_t Size;
  time_t ModTime;
  std::unique_ptr<llvm::MemoryBuffer> Written;
  {
    std::string Contents = "contents\n";
    auto WrittenOrErr = ModCache->write(
        Path, llvm::MemoryBuffer::getMemBufferCopy(Contents, Path), Size,
        ModTime);
    ASSERT_EQ(WrittenOrErr.getError(), std::error_code{});
    Written = std::move(*WrittenOrErr);
  }
  // The returned buffer doesn't reference the written contents.
  ASSERT_TRUE(Written);
  EXPECT_EQ(Written->getBuffer(), "contents\n");

  // Reads share the memory of the written buffer.
  std::unique_ptr<llvm::MemoryBuffer> Read;
  ASSERT_THAT_ERROR(ModCache->read(Path, Size, ModTime).moveInto(Read),
                    llvm::Succeeded());
  EXPECT_EQ(Read->getBufferStart(), Written->getBufferStart());

  // Writing the same PCM again shares the memory of the first write.
  auto RewrittenOrErr = ModCache->write(
      Path, llvm::MemoryBuffer::getMemBufferCopy("contents\n", Path), Size,
      ModTime);
  ASSERT_EQ(RewrittenOrErr.getError(), std::error_code{});
  ASSERT_TRUE(*RewrittenOrErr);
  EXPECT_EQ((*RewrittenOrErr)->getBufferStart(), Written->getBufferStart());
}
