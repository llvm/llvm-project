//===-- unittests/Runtime/MemoryMap.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for the optional read-only-destination copy-out feature
// (FLANG_RT_COPYOUT_READONLY_MODE). The parser, span, and protection
// classifiers are tested directly; the behavioral arms run in death-test
// subprocesses because the mode and the memory-map snapshot are latched once
// per process.

#include "CrashHandlerFixture.h"
#include "tools.h"
#include "gtest/gtest.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/memory-map.h"
#include "flang/Runtime/assign.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace Fortran::runtime;
using namespace Fortran::runtime::memmap;
using Fortran::common::TypeCategory;

//===----------------------------------------------------------------------===//
// ParseProcMaps: filtering and coalescing
//===----------------------------------------------------------------------===//

static bool Parse(const std::string &s, bool fileBackedOnly, Region **out,
    std::size_t *count) {
  return ParseProcMaps(s.data(), s.size(), fileBackedOnly, out, count);
}

TEST(MemoryMapParse, FiltersAndCoalesces) {
  std::string maps{// kept (file-backed private RO)
      "1000-2000 r--p 00000000 08:01 41 /lib/a.so\n"
      // kept and coalesced with the previous entry (contiguous)
      "2000-3000 r-xp 00001000 08:01 41 /lib/a.so\n"
      // rejected: writable
      "3000-4000 rw-p 00002000 08:01 41 /lib/a.so\n"
      // rejected in trust mode: anonymous (inode 0, no path)
      "5000-6000 r--p 00000000 00:00 0 \n"
      // rejected in trust mode: shared mapping
      "6000-7000 r--s 00000000 08:01 42 /lib/b.so\n"
      // rejected in trust mode: deleted file
      "7000-8000 r--p 00000000 08:01 43 /lib/c.so (deleted)\n"
      // rejected in trust mode: pseudo-mapping
      "8000-9000 r--p 00000000 00:00 0 [vvar]\n"
      // kept (second disjoint entry)
      "a000-b000 r--p 00000000 08:01 44 /lib/d.so\n"};
  Region *regions{nullptr};
  std::size_t count{0};
  ASSERT_TRUE(Parse(maps, /*fileBackedOnly=*/true, &regions, &count));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(regions[0].start, 0x1000u);
  EXPECT_EQ(regions[0].end, 0x3000u); // coalesced r--p + r-xp
  EXPECT_EQ(regions[1].start, 0xa000u);
  EXPECT_EQ(regions[1].end, 0xb000u);
  std::free(regions);

  // Permissive filter (confirm mode) also keeps anonymous/shared/deleted
  // read-only entries.
  ASSERT_TRUE(Parse(maps, /*fileBackedOnly=*/false, &regions, &count));
  ASSERT_EQ(count, 3u);
  EXPECT_EQ(regions[1].start, 0x5000u);
  EXPECT_EQ(regions[1].end, 0x9000u); // anon+shared+deleted+vvar coalesced
  EXPECT_EQ(regions[2].start, 0xa000u);
  std::free(regions);
}

TEST(MemoryMapParse, FailClosedOnAnomalies) {
  Region *regions{nullptr};
  std::size_t count{0};
  const char *bad[]{
      "1000-2000 r--p 00000000 08:01 41 /lib/a.so", // no trailing newline
      "2000-1000 r--p 00000000 08:01 41 /lib/a.so\n", // end <= start
      "1000-2000 q--p 00000000 08:01 41 /lib/a.so\n", // bad perm char
      "1000-2000 r--p 00000000 0801 41 /lib/a.so\n", // malformed dev field
      "1000-2000 r--p 00000000 08:01 x /lib/a.so\n", // non-numeric inode
      "zzzz-2000 r--p 00000000 08:01 41 /lib/a.so\n", // bad hex
      "1000-2000\n", // truncated fields
      // out of order
      "2000-3000 r--p 00000000 08:01 41 /a\n"
      "1000-1800 r--p 00000000 08:01 41 /a\n",
      // overlapping
      "1000-3000 r--p 00000000 08:01 41 /a\n"
      "2000-4000 r--p 00000000 08:01 41 /a\n",
  };
  for (const char *entry : bad) {
    std::string s{entry};
    EXPECT_FALSE(Parse(s, true, &regions, &count)) << "input: " << entry;
    EXPECT_EQ(regions, nullptr) << "input: " << entry;
  }
  // Anomalies after an acceptable prefix must not publish the prefix.
  std::string prefixThenBad{"1000-2000 r--p 00000000 08:01 41 /lib/a.so\n"
                            "3000-2800 r--p 00000000 08:01 41 /lib/a.so\n"};
  EXPECT_FALSE(Parse(prefixThenBad, true, &regions, &count));
  EXPECT_EQ(regions, nullptr);
}

//===----------------------------------------------------------------------===//
// SpanIsContained
//===----------------------------------------------------------------------===//

TEST(MemoryMapSpan, Containment) {
  Region regions[]{{0x1000, 0x3000}, {0x5000, 0x6000}};
  EXPECT_TRUE(SpanIsContained(0x1000, 0x3000, regions, 2));
  EXPECT_TRUE(SpanIsContained(0x1800, 0x2800, regions, 2));
  EXPECT_TRUE(SpanIsContained(0x5fff, 0x6000, regions, 2));
  EXPECT_FALSE(SpanIsContained(0x0fff, 0x2000, regions, 2)); // starts before
  EXPECT_FALSE(SpanIsContained(0x2000, 0x3001, regions, 2)); // ends after
  EXPECT_FALSE(SpanIsContained(0x3000, 0x5000, regions, 2)); // gap
  EXPECT_FALSE(SpanIsContained(0x4000, 0x4800, regions, 2)); // hole
  EXPECT_FALSE(SpanIsContained(0x2000, 0x2000, regions, 2)); // empty span
  EXPECT_FALSE(SpanIsContained(0x1000, 0x2000, nullptr, 0)); // empty table
}

//===----------------------------------------------------------------------===//
// ComputeDataSpan
//===----------------------------------------------------------------------===//

TEST(MemoryMapSpan, DescriptorSpans) {
  auto array{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{8},
      std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8}, sizeof(std::int32_t))};
  std::uintptr_t lo{0}, hi{0};
  ASSERT_TRUE(ComputeDataSpan(*array, lo, hi));
  auto base{reinterpret_cast<std::uintptr_t>(array->OffsetElement<char>())};
  EXPECT_EQ(lo, base);
  EXPECT_EQ(hi, base + 8 * sizeof(std::int32_t));

  // Strided view: elements 1,3,5,7 - span still covers first..last touched.
  array->GetDimension(0).SetByteStride(2 * sizeof(std::int32_t));
  array->GetDimension(0).SetExtent(4);
  ASSERT_TRUE(ComputeDataSpan(*array, lo, hi));
  EXPECT_EQ(lo, base);
  EXPECT_EQ(hi, base + 6 * sizeof(std::int32_t) + sizeof(std::int32_t));

  // Negative stride: base points at the LAST touched element.
  array->set_base_addr(array->OffsetElement<char>(6 * sizeof(std::int32_t)));
  array->GetDimension(0).SetByteStride(
      -2 * static_cast<std::int64_t>(sizeof(std::int32_t)));
  ASSERT_TRUE(ComputeDataSpan(*array, lo, hi));
  EXPECT_EQ(lo, base);
  EXPECT_EQ(hi, base + 7 * sizeof(std::int32_t));

  // Zero extent => false.
  array->GetDimension(0).SetExtent(0);
  EXPECT_FALSE(ComputeDataSpan(*array, lo, hi));
}

//===----------------------------------------------------------------------===//
// Windows protection classification (pure function; runs on any host)
//===----------------------------------------------------------------------===//

TEST(MemoryMapWindows, ProtectionClassification) {
  constexpr std::uint32_t commit{0x1000}, reserve{0x2000}, image{0x1000000},
      priv{0x20000};
  constexpr std::uint32_t ro{0x02}, rw{0x04}, wc{0x08}, xr{0x20}, xwc{0x80},
      guard{0x100}, nocache{0x200};
  // Accepted: committed image PAGE_READONLY / PAGE_EXECUTE_READ.
  EXPECT_TRUE(ProtectionIsReadOnly(commit, ro, image, true));
  EXPECT_TRUE(ProtectionIsReadOnly(commit, xr, image, true));
  // Write-copy protections are writable-on-fault: never read-only.
  EXPECT_FALSE(ProtectionIsReadOnly(commit, wc, image, true));
  EXPECT_FALSE(ProtectionIsReadOnly(commit, xwc, image, true));
  // Modifier bits (guard, nocache) reject the region outright.
  EXPECT_FALSE(ProtectionIsReadOnly(commit, ro | guard, image, true));
  EXPECT_FALSE(ProtectionIsReadOnly(commit, ro | nocache, image, true));
  // Writable / no-access / reserved states.
  EXPECT_FALSE(ProtectionIsReadOnly(commit, rw, image, true));
  EXPECT_FALSE(ProtectionIsReadOnly(reserve, ro, image, true));
  // imageOnly (trust table) rejects private RO; permissive mode keeps it.
  EXPECT_FALSE(ProtectionIsReadOnly(commit, ro, priv, true));
  EXPECT_TRUE(ProtectionIsReadOnly(commit, ro, priv, false));
}

//===----------------------------------------------------------------------===//
// Behavioral arms (Linux; each runs in a fresh subprocess because the mode
// and the snapshot latch once per process)
//===----------------------------------------------------------------------===//

#if defined(__linux__)

namespace {
// A file-backed private read-only mapping of one page holding 'count' int32
// values - the shape the trust-mode table keeps (inode != 0, private, r--).
struct RoFileMapping {
  static constexpr std::size_t count{16};
  std::int32_t *data{nullptr};
  std::size_t pageSize{0};

  bool Map(bool readOnlyNow) {
    pageSize = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    // The file must stay linked while mapped: an unlinked-but-mapped file
    // shows as "(deleted)" in /proc/self/maps, which the trust-mode filter
    // rejects. Unlink happens in the destructor.
    std::strcpy(path_, "copyout-romap-XXXXXX");
    int fd{::mkstemp(path_)};
    if (fd < 0) {
      return false;
    }
    std::vector<std::int32_t> initial(pageSize / sizeof(std::int32_t));
    for (std::size_t i{0}; i < initial.size(); ++i) {
      initial[i] = static_cast<std::int32_t>(i + 1);
    }
    if (::write(fd, initial.data(), pageSize) !=
        static_cast<::ssize_t>(pageSize)) {
      ::close(fd);
      return false;
    }
    void *p{
        ::mmap(nullptr, pageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)};
    ::close(fd);
    if (p == MAP_FAILED) {
      return false;
    }
    if (readOnlyNow && ::mprotect(p, pageSize, PROT_READ) != 0) {
      return false;
    }
    data = static_cast<std::int32_t *>(p);
    return true;
  }
  ~RoFileMapping() {
    if (data) {
      ::munmap(data, pageSize);
    }
    if (path_[0]) {
      ::unlink(path_);
    }
  }
  char path_[32]{};
};

} // namespace

// Headline: a MODIFIED temporary copied out into read-only storage does not
// fault and stores nothing in mode 1 - the exact case that faults with the
// feature off.
TEST(MemoryMapCopyOut, Mode1SkipsModifiedTempIntoReadOnly) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::setenv("FLANG_RT_COPYOUT_READONLY_MODE", "1", 1);
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/true)) {
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        *temp.OffsetElement<std::int32_t>(0) = 999; // invalid-program write
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__);
        // Reaching here means no fault; verify nothing was stored.
        _exit(m.data[0] == 1 ? 0 : 3);
      },
      testing::ExitedWithCode(0), "");
}

// Staleness, mode 1: after mprotect RO->RW (post-snapshot), the stale trust
// table still skips - the DOCUMENTED accepted lost write of trust mode.
TEST(MemoryMapCopyOut, Mode1StalenessSkipsAfterReprotect) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::setenv("FLANG_RT_COPYOUT_READONLY_MODE", "1", 1);
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/true)) {
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        // Force the snapshot while the page is read-only.
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__);
        // Now make it writable and try a real copy-out.
        if (::mprotect(m.data, m.pageSize, PROT_READ | PROT_WRITE) != 0) {
          _exit(2);
        }
        StaticDescriptor<1> staticTemp2;
        Descriptor &temp2{staticTemp2.descriptor()};
        RTNAME(CopyInAssign)(temp2, var);
        *temp2.OffsetElement<std::int32_t>(0) = 999;
        RTNAME(CopyOutAssign)(&var, temp2, __FILE__, __LINE__);
        // Trust mode consults the stale table: the write is (documentedly)
        // lost.
        _exit(m.data[0] == 1 ? 0 : 3);
      },
      testing::ExitedWithCode(0), "");
}

// Staleness, mode 2: same scenario, but confirm-on-hit sees the current RW
// protection and the write goes through.
TEST(MemoryMapCopyOut, Mode2ConfirmWritesAfterReprotect) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::setenv("FLANG_RT_COPYOUT_READONLY_MODE", "2", 1);
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/true)) {
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__); // snapshot
        if (::mprotect(m.data, m.pageSize, PROT_READ | PROT_WRITE) != 0) {
          _exit(2);
        }
        StaticDescriptor<1> staticTemp2;
        Descriptor &temp2{staticTemp2.descriptor()};
        RTNAME(CopyInAssign)(temp2, var);
        *temp2.OffsetElement<std::int32_t>(0) = 999;
        RTNAME(CopyOutAssign)(&var, temp2, __FILE__, __LINE__);
        _exit(m.data[0] == 999 ? 0 : 3);
      },
      testing::ExitedWithCode(0), "");
}

// Mode 2 with the page still read-only: candidate confirmed, skip, no fault.
TEST(MemoryMapCopyOut, Mode2SkipsModifiedTempIntoReadOnly) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::setenv("FLANG_RT_COPYOUT_READONLY_MODE", "2", 1);
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/true)) {
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        *temp.OffsetElement<std::int32_t>(0) = 999;
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__);
        _exit(m.data[0] == 1 ? 0 : 3);
      },
      testing::ExitedWithCode(0), "");
}

// Feature off (default): the same modified-temp-into-RO copy-out faults - the
// upstream propagate-and-catch behavior is preserved.
TEST(MemoryMapCopyOut, ModeOffStillFaults) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::unsetenv("FLANG_RT_COPYOUT_READONLY_MODE");
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/true)) {
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        *temp.OffsetElement<std::int32_t>(0) = 999;
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__);
        _exit(0); // not reached
      },
      testing::KilledBySignal(SIGSEGV), "");
}

// Writable destinations round-trip unchanged with the feature on.
TEST(MemoryMapCopyOut, Mode1WritableDestUnaffected) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        ::setenv("FLANG_RT_COPYOUT_READONLY_MODE", "1", 1);
        RoFileMapping m;
        if (!m.Map(/*readOnlyNow=*/false)) { // stays RW
          _exit(2);
        }
        SubscriptValue extent[1]{RoFileMapping::count};
        StaticDescriptor<1> staticVar;
        Descriptor &var{staticVar.descriptor()};
        var.Establish(
            TypeCategory::Integer, sizeof(std::int32_t), m.data, 1, extent);
        StaticDescriptor<1> staticTemp;
        Descriptor &temp{staticTemp.descriptor()};
        RTNAME(CopyInAssign)(temp, var);
        *temp.OffsetElement<std::int32_t>(0) = 999;
        RTNAME(CopyOutAssign)(&var, temp, __FILE__, __LINE__);
        _exit(m.data[0] == 999 ? 0 : 3);
      },
      testing::ExitedWithCode(0), "");
}

#endif // __linux__
