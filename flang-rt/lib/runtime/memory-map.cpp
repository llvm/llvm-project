//===-- lib/runtime/memory-map.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/memory-map.h"
#include "flang-rt/runtime/descriptor.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(RT_DEVICE_COMPILATION) && !defined(RT_GPU_TARGET)

#if defined(_WIN32)
#include "flang/Common/windows-include.h"
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__) && __has_include(<linux/fs.h>)
#include <linux/fs.h> // may define PROCMAP_QUERY (kernel >= 6.11 headers)
#include <sys/ioctl.h>
#endif
#endif

namespace Fortran::runtime {
namespace memmap {

// All memory in this module is plain malloc/realloc/free: an optional feature
// must not crash the program on allocation failure (no AllocateMemoryOrCrash);
// any failure makes the feature permanently inert instead.

//===----------------------------------------------------------------------===//
// Span computation
//===----------------------------------------------------------------------===//

bool ComputeDataSpan(
    const Descriptor &var, std::uintptr_t &lo, std::uintptr_t &hi) {
  const char *base{var.OffsetElement<char>()};
  std::size_t elementBytes{var.ElementBytes()};
  if (!base || elementBytes == 0) {
    return false; // unallocated, or zero-length elements (e.g. CHARACTER(0))
  }
  // Sum negative-stride and positive-stride reaches separately so that the
  // span is correct for any stride signs. All arithmetic is overflow-checked;
  // no out-of-object C++ pointer arithmetic is performed.
  std::int64_t negReach{0}; // <= 0
  std::int64_t posReach{0}; // >= 0
  for (int j{0}; j < var.rank(); ++j) {
    const auto &dim{var.GetDimension(j)};
    std::int64_t extent{dim.Extent()};
    if (extent <= 0) {
      return false; // zero-sized array: nothing will be stored anyway
    }
    std::int64_t stride{dim.ByteStride()};
    std::int64_t reach;
    if (__builtin_mul_overflow(extent - 1, stride, &reach)) {
      return false;
    }
    if (reach < 0) {
      if (__builtin_add_overflow(negReach, reach, &negReach)) {
        return false;
      }
    } else if (__builtin_add_overflow(posReach, reach, &posReach)) {
      return false;
    }
  }
  auto baseAddr{reinterpret_cast<std::uintptr_t>(base)};
  std::uintptr_t loAddr, hiAddr;
  if (negReach < 0) {
    std::uintptr_t down{static_cast<std::uintptr_t>(-negReach)};
    if (down > baseAddr) {
      return false;
    }
    loAddr = baseAddr - down;
  } else {
    loAddr = baseAddr;
  }
  if (__builtin_add_overflow(
          baseAddr, static_cast<std::uintptr_t>(posReach), &hiAddr) ||
      __builtin_add_overflow(hiAddr, elementBytes, &hiAddr)) {
    return false;
  }
  if (hiAddr <= loAddr) {
    return false;
  }
  lo = loAddr;
  hi = hiAddr;
  return true;
}

//===----------------------------------------------------------------------===//
// Region table and containment
//===----------------------------------------------------------------------===//

bool SpanIsContained(std::uintptr_t lo, std::uintptr_t hi,
    const Region *regions, std::size_t count) {
  if (!regions || count == 0 || hi <= lo) {
    return false;
  }
  // Binary search: last region with start <= lo. Regions are ascending and
  // coalesced, so containment must be within that single region.
  std::size_t first{0}, n{count};
  while (n > 1) {
    std::size_t half{n / 2};
    if (regions[first + half].start <= lo) {
      first += half;
      n -= half;
    } else {
      n = half;
    }
  }
  return regions[first].start <= lo && hi <= regions[first].end;
}

// Fixed-capacity guard against a pathological number of mappings; a process
// with more read-only regions than this simply gets an inert feature.
static constexpr std::size_t maxRegions{65536};

namespace {
struct RegionBuilder {
  Region *data{nullptr};
  std::size_t size{0};
  std::size_t capacity{0};
  std::uintptr_t lastEnd{0}; // monotonicity watermark over ALL parsed entries

  ~RegionBuilder() { std::free(data); }

  // Appends a kept region, coalescing with the previous kept one when
  // contiguous. Returns false on allocation failure or capacity exhaustion.
  bool Append(std::uintptr_t start, std::uintptr_t end) {
    if (size > 0 && data[size - 1].end == start) {
      data[size - 1].end = end;
      return true;
    }
    if (size == capacity) {
      if (capacity >= maxRegions) {
        return false;
      }
      std::size_t newCap{capacity ? capacity * 2 : 64};
      void *p{std::realloc(data, newCap * sizeof(Region))};
      if (!p) {
        return false;
      }
      data = static_cast<Region *>(p);
      capacity = newCap;
    }
    data[size++] = Region{start, end};
    return true;
  }

  Region *Release(std::size_t &countOut) {
    Region *result{data};
    countOut = size;
    data = nullptr;
    size = capacity = 0;
    return result;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linux /proc/self/maps parsing (fail-closed)
//===----------------------------------------------------------------------===//

static bool ParseHex(const char *&p, const char *end, std::uintptr_t &value) {
  std::uintptr_t v{0};
  const char *start{p};
  while (p < end) {
    char c{*p};
    unsigned digit;
    if (c >= '0' && c <= '9') {
      digit = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      digit = c - 'a' + 10;
    } else {
      break;
    }
    if (v > (~static_cast<std::uintptr_t>(0)) >> 4) {
      return false; // overflow
    }
    v = (v << 4) | digit;
    ++p;
  }
  if (p == start) {
    return false;
  }
  value = v;
  return true;
}

static bool ParseDec(const char *&p, const char *end, std::uint64_t &value) {
  std::uint64_t v{0};
  const char *start{p};
  while (p < end && *p >= '0' && *p <= '9') {
    if (v > (UINT64_MAX - 9) / 10) {
      return false;
    }
    v = v * 10 + (*p - '0');
    ++p;
  }
  if (p == start) {
    return false;
  }
  value = v;
  return true;
}

bool ParseProcMaps(const char *buf, std::size_t len, bool fileBackedOnly,
    Region **out, std::size_t *outCount) {
  *out = nullptr;
  *outCount = 0;
  RegionBuilder builder;
  const char *p{buf};
  const char *end{buf + len};
  while (p < end) {
    const char *lineEnd{static_cast<const char *>(
        std::memchr(p, '\n', static_cast<std::size_t>(end - p)))};
    if (!lineEnd) {
      return false; // truncated final line: fail closed, publish nothing
    }
    // <start>-<end> <perms> <offset> <dev> <inode> [path]
    std::uintptr_t start, stop;
    if (!ParseHex(p, lineEnd, start) || p >= lineEnd || *p++ != '-' ||
        !ParseHex(p, lineEnd, stop) || p >= lineEnd || *p++ != ' ') {
      return false;
    }
    if (stop <= start || start < builder.lastEnd) {
      return false; // empty, out-of-order, or overlapping entry
    }
    builder.lastEnd = stop;
    if (lineEnd - p < 5) {
      return false;
    }
    char permR{p[0]}, permW{p[1]}, permX{p[2]}, permP{p[3]};
    if ((permR != 'r' && permR != '-') || (permW != 'w' && permW != '-') ||
        (permX != 'x' && permX != '-') || (permP != 'p' && permP != 's')) {
      return false;
    }
    p += 4;
    if (*p++ != ' ') {
      return false;
    }
    std::uintptr_t offset;
    if (!ParseHex(p, lineEnd, offset) || p >= lineEnd || *p++ != ' ') {
      return false;
    }
    std::uintptr_t devMajor, devMinor;
    if (!ParseHex(p, lineEnd, devMajor) || p >= lineEnd || *p++ != ':' ||
        !ParseHex(p, lineEnd, devMinor) || p >= lineEnd || *p++ != ' ') {
      return false;
    }
    std::uint64_t inode;
    if (!ParseDec(p, lineEnd, inode)) {
      return false;
    }
    while (p < lineEnd && *p == ' ') {
      ++p;
    }
    const char *path{p};
    std::size_t pathLen{static_cast<std::size_t>(lineEnd - p)};
    // Keep only readable, non-writable mappings. Both modes require 'r':
    // a destination that was copied in from cannot have been PROT_NONE or
    // execute-only, so its absence signals a bug, not a constant.
    bool keep{permR == 'r' && permW == '-'};
    if (keep && fileBackedOnly) {
      // Trust-table filter: file-backed private mappings only. Anonymous
      // read-only pages are the most recycling-prone (mprotect'd arenas,
      // JIT) and contribute nothing to the PARAMETER/.rodata target.
      keep = permP == 'p' && inode != 0 && pathLen > 0 && path[0] == '/' &&
          !(pathLen >= 9 &&
              std::memcmp(path + pathLen - 9, "(deleted)", 9) == 0);
    }
    if (keep && !builder.Append(start, stop)) {
      return false;
    }
    p = lineEnd + 1;
  }
  *out = builder.Release(*outCount);
  return true;
}

//===----------------------------------------------------------------------===//
// Windows protection classification (compiled everywhere for testability)
//===----------------------------------------------------------------------===//

// Local mirrors of the Windows constants so this classifier can be unit-tested
// on any host. Values are fixed ABI constants.
static constexpr std::uint32_t kMemCommit{0x1000};
static constexpr std::uint32_t kMemImage{0x1000000};
static constexpr std::uint32_t kPageReadonly{0x02};
static constexpr std::uint32_t kPageExecuteRead{0x20};

bool ProtectionIsReadOnly(std::uint32_t state, std::uint32_t protect,
    std::uint32_t type, bool imageOnly) {
  if (state != kMemCommit) {
    return false;
  }
  if (imageOnly && type != kMemImage) {
    return false;
  }
  // Exactly PAGE_READONLY or PAGE_EXECUTE_READ, with no modifier bits at all:
  // this rejects PAGE_GUARD, PAGE_NOCACHE, PAGE_WRITECOMBINE, the write-copy
  // protections (writable on fault), and every unknown combination.
  return protect == kPageReadonly || protect == kPageExecuteRead;
}

//===----------------------------------------------------------------------===//
// Snapshot enumeration
//===----------------------------------------------------------------------===//

#if defined(_WIN32)

static bool EnumerateReadOnlyRegions(
    bool imageOnly, Region **out, std::size_t *outCount) {
  *out = nullptr;
  *outCount = 0;
  RegionBuilder builder;
  SYSTEM_INFO si;
  GetNativeSystemInfo(&si);
  std::uintptr_t address{0};
  std::uintptr_t maxAddress{
      reinterpret_cast<std::uintptr_t>(si.lpMaximumApplicationAddress)};
  while (address < maxAddress) {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(reinterpret_cast<LPCVOID>(address), &mbi, sizeof mbi) ==
        0) {
      break;
    }
    std::uintptr_t regionBase{
        reinterpret_cast<std::uintptr_t>(mbi.BaseAddress)};
    std::uintptr_t regionEnd;
    if (__builtin_add_overflow(regionBase, mbi.RegionSize, &regionEnd) ||
        regionEnd <= address) {
      return false; // wrap or no forward progress
    }
    if (ProtectionIsReadOnly(mbi.State, mbi.Protect, mbi.Type, imageOnly) &&
        !builder.Append(regionBase, regionEnd)) {
      return false;
    }
    address = regionEnd;
  }
  *out = builder.Release(*outCount);
  return true;
}

// Confirms [lo, hi) is currently committed read-only. VirtualQuery is the
// native per-range primitive; no table is involved.
static bool CurrentlyReadOnly(std::uintptr_t lo, std::uintptr_t hi) {
  std::uintptr_t address{lo};
  while (address < hi) {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(reinterpret_cast<LPCVOID>(address), &mbi, sizeof mbi) ==
        0) {
      return false;
    }
    if (!ProtectionIsReadOnly(
            mbi.State, mbi.Protect, mbi.Type, /*imageOnly=*/false)) {
      return false;
    }
    std::uintptr_t regionEnd;
    if (__builtin_add_overflow(
            reinterpret_cast<std::uintptr_t>(mbi.BaseAddress), mbi.RegionSize,
            &regionEnd) ||
        regionEnd <= address) {
      return false;
    }
    address = regionEnd;
  }
  return true;
}

#elif defined(__linux__)

// Reads all of /proc/self/maps into a malloc'd buffer. procfs files are not
// seekable or stat-able for size, so read in a doubling loop.
static char *ReadWholeProcMaps(std::size_t *lenOut) {
  int fd{-1};
  do {
    fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return nullptr;
  }
  std::size_t capacity{1u << 16};
  std::size_t length{0};
  char *buffer{static_cast<char *>(std::malloc(capacity))};
  while (buffer) {
    if (length == capacity) {
      if (capacity >= (1u << 26)) { // 64 MiB cap: fail closed
        break;
      }
      capacity *= 2;
      void *p{std::realloc(buffer, capacity)};
      if (!p) {
        break;
      }
      buffer = static_cast<char *>(p);
    }
    ::ssize_t n{::read(fd, buffer + length, capacity - length)};
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (n == 0) {
      ::close(fd);
      *lenOut = length;
      return buffer;
    }
    length += static_cast<std::size_t>(n);
  }
  ::close(fd);
  std::free(buffer);
  return nullptr;
}

static bool EnumerateReadOnlyRegions(
    bool fileBackedOnly, Region **out, std::size_t *outCount) {
  std::size_t length;
  char *buffer{ReadWholeProcMaps(&length)};
  if (!buffer) {
    return false;
  }
  bool ok{ParseProcMaps(buffer, length, fileBackedOnly, out, outCount)};
  std::free(buffer);
  return ok;
}

#if defined(PROCMAP_QUERY)
// Linux >= 6.11: per-VMA query ioctl on /proc/self/maps - no full traversal.
// Returns 1 = read-only over the span, 0 = not, -1 = unsupported (fall back).
static int QuerySpanReadOnlyIoctl(std::uintptr_t lo, std::uintptr_t hi) {
  int fd{-1};
  do {
    fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return -1;
  }
  std::uintptr_t address{lo};
  while (address < hi) {
    struct procmap_query q;
    std::memset(&q, 0, sizeof q);
    q.size = sizeof q;
    q.query_flags = 0; // covering VMA only
    q.query_addr = address;
    if (::ioctl(fd, PROCMAP_QUERY, &q) < 0) {
      int e{errno};
      ::close(fd);
      return (e == ENOTTY || e == EINVAL || e == EOPNOTSUPP) ? -1 : 0;
    }
    if (!(q.vma_flags & PROCMAP_QUERY_VMA_READABLE) ||
        (q.vma_flags & PROCMAP_QUERY_VMA_WRITABLE) || q.vma_end <= address) {
      ::close(fd);
      return 0;
    }
    address = q.vma_end;
  }
  ::close(fd);
  return 1;
}
#endif // PROCMAP_QUERY

static bool CurrentlyReadOnly(std::uintptr_t lo, std::uintptr_t hi) {
#if defined(PROCMAP_QUERY)
  if (int r{QuerySpanReadOnlyIoctl(lo, hi)}; r >= 0) {
    return r == 1;
  }
#endif
  // Fallback: re-read and re-parse the current map (any mapping type; the
  // question here is current permissions, not provenance).
  Region *regions{nullptr};
  std::size_t count{0};
  if (!EnumerateReadOnlyRegions(/*fileBackedOnly=*/false, &regions, &count)) {
    return false;
  }
  bool result{SpanIsContained(lo, hi, regions, count)};
  std::free(regions);
  return result;
}

#else // neither _WIN32 nor __linux__: feature inert

static bool EnumerateReadOnlyRegions(bool, Region **out, std::size_t *count) {
  *out = nullptr;
  *count = 0;
  return false;
}
static bool CurrentlyReadOnly(std::uintptr_t, std::uintptr_t) { return false; }

#endif

//===----------------------------------------------------------------------===//
// Lazy, fail-closed, non-blocking snapshot state machine
//===----------------------------------------------------------------------===//

enum : int { kUninitialized = 0, kBuilding = 1, kReady = 2, kInert = 3 };

static std::atomic<int> tableState{kUninitialized};
// One immutable table per process lifetime, intentionally leaked, never
// replaced or reclaimed. Published only by the release store to tableState.
static Region *roTable{nullptr};
static std::size_t roTableCount{0};

static bool EnsureSnapshot() {
  int s{tableState.load(std::memory_order_acquire)};
  if (s == kReady) {
    return true;
  }
  if (s != kUninitialized) {
    return false; // Building (never block; a fork child inheriting Building
                  // observes exactly this) or Inert
  }
  int expected{kUninitialized};
  if (!tableState.compare_exchange_strong(expected, kBuilding,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return tableState.load(std::memory_order_acquire) == kReady;
  }
  Region *regions{nullptr};
  std::size_t count{0};
  if (EnumerateReadOnlyRegions(/*fileBackedOnly=*/true, &regions, &count)) {
    roTable = regions;
    roTableCount = count;
    tableState.store(kReady, std::memory_order_release);
    return true;
  }
  tableState.store(kInert, std::memory_order_release);
  return false;
}

} // namespace memmap

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

CopyOutReadOnlyMode GetCopyOutReadOnlyMode() {
  // Read lazily and independently of ExecutionEnvironment::Configure, which
  // never runs under a non-Fortran main program.
  static std::atomic<int> cached{-1};
  int mode{cached.load(std::memory_order_relaxed)};
  if (mode < 0) {
    mode = 0;
    if (const char *x{std::getenv("FLANG_RT_COPYOUT_READONLY_MODE")}) {
      char *end;
      long n{std::strtol(x, &end, 10)};
      if (n >= 0 && n <= 2 && *end == '\0' && end != x) {
        mode = static_cast<int>(n);
      } // anything else fails closed to Off
    }
    cached.store(mode, std::memory_order_relaxed);
  }
  return static_cast<CopyOutReadOnlyMode>(mode);
}

bool CopyOutReadOnlyCandidate(const Descriptor &var) {
  std::uintptr_t lo, hi;
  if (!memmap::ComputeDataSpan(var, lo, hi)) {
    return false;
  }
  if (!memmap::EnsureSnapshot()) {
    return false;
  }
  return memmap::SpanIsContained(lo, hi, memmap::roTable, memmap::roTableCount);
}

bool CopyOutReadOnlyConfirm(const Descriptor &var) {
  std::uintptr_t lo, hi;
  if (!memmap::ComputeDataSpan(var, lo, hi)) {
    return false;
  }
  return memmap::CurrentlyReadOnly(lo, hi);
}

void NoteSkippedCopyOut(const char *sourceFile, int sourceLine) {
  static std::atomic<std::uint64_t> skipCount{0};
  static std::atomic<int> diagEnabled{-1};
  std::uint64_t n{skipCount.fetch_add(1, std::memory_order_relaxed) + 1};
  int enabled{diagEnabled.load(std::memory_order_relaxed)};
  if (enabled < 0) {
    const char *x{std::getenv("FLANG_RT_COPYOUT_READONLY_DIAG")};
    enabled = x && x[0] == '1' && x[1] == '\0';
    diagEnabled.store(enabled, std::memory_order_relaxed);
  }
  if (enabled && n <= 10) {
    std::fprintf(stderr,
        "flang-rt: skipped copy-out to read-only memory (%s:%d) [%llu]\n",
        sourceFile ? sourceFile : "<unknown>", sourceLine,
        static_cast<unsigned long long>(n));
  }
}

} // namespace Fortran::runtime

#endif // !RT_DEVICE_COMPILATION && !RT_GPU_TARGET
