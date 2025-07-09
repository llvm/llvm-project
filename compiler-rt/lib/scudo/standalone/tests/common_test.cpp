//===-- common_test.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal_defs.h"
#include "tests/scudo_unit_test.h"

#include "common.h"
#include "mem_map.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <string>

namespace scudo {

static void GetRssKbFromString(uptr MapAddress, std::string &Buffer,
                               size_t &ParsedBytes, size_t &RssKb,
                               bool &Found) {
  size_t LineStart = 0;
  bool FindRss = false;
  while (true) {
    size_t LineEnd = Buffer.find('\n', LineStart);
    if (LineEnd == std::string::npos) {
      ParsedBytes = LineStart;
      ASSERT_NE(0U, ParsedBytes)
          << "The current buffer size (" << Buffer.size()
          << ") is not large enough to contain a single line.";
      break;
    }
    Buffer[LineEnd] = '\0';
    // The format of the address line is:
    //   55ecba642000-55ecba644000 r--p 00000000 fd:01 66856632
    uptr StartAddr;
    uptr EndAddr;
    char Perms[5];
    if (sscanf(&Buffer[LineStart], "%" SCNxPTR "-%" SCNxPTR " %4s", &StartAddr,
               &EndAddr, Perms) == 3) {
      if (StartAddr == MapAddress) {
        FindRss = true;
      }
    } else if (FindRss && strncmp(&Buffer[LineStart], "Rss:", 4) == 0) {
      // The format of the RSS line is:
      //   Rss:                   8 kB
      ASSERT_EQ(1, sscanf(&Buffer[LineStart], "Rss: %zd kB", &RssKb))
          << "Bad Rss Line: " << &Buffer[LineStart];
      Found = true;
      ParsedBytes = LineStart;
      break;
    }
    LineStart = LineEnd + 1;
  }
}

static void GetRssKb(void *BaseAddress, size_t &RssKb) {
  if (!SCUDO_LINUX)
    UNREACHABLE("Not implemented!");

  size_t MapAddress = reinterpret_cast<size_t>(BaseAddress);

  int Fd = open("/proc/self/smaps", O_RDONLY);
  ASSERT_NE(-1, Fd) << "Failed to open /proc/self/smaps: " << strerror(errno);

  std::string Buffer(10 * 1024, '\0');
  size_t LeftoverBytes = 0;
  RssKb = 0;
  bool FoundMap = false;
  while (LeftoverBytes != Buffer.size()) {
    ssize_t ReadBytes =
        read(Fd, &Buffer[LeftoverBytes], Buffer.size() - LeftoverBytes);
    if (ReadBytes < 0) {
      EXPECT_GT(0, ReadBytes) << "read failed: " << strerror(errno);
      break;
    }
    if (ReadBytes == 0) {
      // Nothing left to read.
      break;
    }
    size_t End = static_cast<size_t>(ReadBytes) + LeftoverBytes;
    Buffer[End] = '\0';
    size_t ParsedBytes = 0;
    GetRssKbFromString(MapAddress, Buffer, ParsedBytes, RssKb, FoundMap);
    if (TEST_HAS_FAILURE || FoundMap)
      break;
    // Need to copy the leftover bytes back to the front of the buffer.
    LeftoverBytes = End - ParsedBytes;
    if (LeftoverBytes != 0) {
      memmove(Buffer.data(), &Buffer[ParsedBytes], LeftoverBytes);
    }
  }
  close(Fd);

  EXPECT_TRUE(FoundMap) << "Could not find map at address " << BaseAddress;
}

// Fuchsia needs getRssKb implementation.
TEST(ScudoCommonTest, SKIP_ON_FUCHSIA(ResidentMemorySize)) {
  // Make sure to have the size of the map on a page boundary.
  const uptr PageSize = getPageSizeCached();
  const uptr SizeBytes = 1000 * PageSize;
  const uptr ActualSizeBytes = SizeBytes - 2 * PageSize;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  // Mark the first page and the last page as unreadable to make sure that
  // the map shows up as distinct from all other maps.
  EXPECT_EQ(0, mprotect(reinterpret_cast<void *>(MemMap.getBase()), PageSize,
                        PROT_NONE));
  EXPECT_EQ(0, mprotect(reinterpret_cast<void *>(MemMap.getBase() + SizeBytes -
                                                 PageSize),
                        PageSize, PROT_NONE));

  size_t RssKb = 0;
  void *P = reinterpret_cast<void *>(MemMap.getBase() + PageSize);
  GetRssKb(P, RssKb);
  EXPECT_EQ(RssKb, 0U);

  // Make the entire map resident.
  memset(P, 1, ActualSizeBytes);
  GetRssKb(P, RssKb);
  EXPECT_EQ(RssKb, (ActualSizeBytes >> 10));

  // Should release the memory to the kernel immediately.
  MemMap.releasePagesToOS(MemMap.getBase(), SizeBytes);
  GetRssKb(P, RssKb);
  EXPECT_EQ(RssKb, 0U);

  // Make the entire map resident again.
  memset(P, 1, ActualSizeBytes);
  GetRssKb(P, RssKb);
  EXPECT_EQ(RssKb, ActualSizeBytes >> 10);

  MemMap.unmap();
}

TEST(ScudoCommonTest, Zeros) {
  const uptr Size = 1ull << 20;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, Size, "Zeros"));
  ASSERT_NE(MemMap.getBase(), 0U);
  uptr *P = reinterpret_cast<uptr *>(MemMap.getBase());
  const ptrdiff_t N = Size / sizeof(uptr);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  memset(P, 1, Size);
  EXPECT_EQ(std::count(P, P + N, 0), 0);

  MemMap.releasePagesToOS(MemMap.getBase(), Size);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  MemMap.unmap();
}

} // namespace scudo
