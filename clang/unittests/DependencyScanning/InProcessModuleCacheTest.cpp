#include "clang/DependencyScanning/InProcessModuleCache.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace clang::dependencies;

TEST(InProcessModuleCache, ReadWriteInvalidation) {
  ModuleCacheEntries Entries;
  std::shared_ptr<ModuleCache> ModCache = makeInProcessModuleCache(Entries);

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
  ASSERT_EQ(ModCache->write(Path, *NewBuf, Size, ModTime), std::error_code{});

  // Writing a new buffer should not invalidate the previously-read buffer.
  EXPECT_EQ(Buf->getBuffer(), "original\n");
}

TEST(InProcessModuleCache, ReadReadInvalidation) {
  ModuleCacheEntries Entries;
  std::shared_ptr<ModuleCache> ModCache = makeInProcessModuleCache(Entries);

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
