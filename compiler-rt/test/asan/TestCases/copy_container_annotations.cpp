// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_copy_contiguous_container_annotations.

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr size_t kGranularity = 8;

template <class T> static constexpr T RoundDown(T x) {
  return reinterpret_cast<T>(reinterpret_cast<uintptr_t>(x) &
                             ~(kGranularity - 1));
}
template <class T> static constexpr T RoundUp(T x) {
  return reinterpret_cast<T>(
      RoundDown(reinterpret_cast<uintptr_t>(x) + kGranularity - 1));
}

static std::vector<int> GetPoisonedState(char *begin, char *end) {
  std::vector<int> result;
  for (char *ptr = begin; ptr != end; ++ptr) {
    result.push_back(__asan_address_is_poisoned(ptr));
  }
  return result;
}

static void RandomPoison(char *beg, char *end) {
  assert(beg == RoundDown(beg));
  assert(end == RoundDown(end));
  __asan_poison_memory_region(beg, end - beg);
  for (beg = RoundUp(beg); beg < end; beg += kGranularity) {
    __asan_unpoison_memory_region(beg, rand() % (kGranularity + 1));
  }
}

template <bool benchmark>
static void Test(size_t capacity, size_t off_src, size_t off_dst,
                 char *src_buffer_beg, char *src_buffer_end,
                 char *dst_buffer_beg, char *dst_buffer_end) {
  size_t dst_buffer_size = dst_buffer_end - dst_buffer_beg;
  char *src_beg = src_buffer_beg + off_src;
  char *src_end = src_beg + capacity;

  char *dst_beg = dst_buffer_beg + off_dst;
  char *dst_end = dst_beg + capacity;
  if (benchmark) {
    __sanitizer_copy_contiguous_container_annotations(src_beg, src_end, dst_beg,
                                                      dst_end);
    return;
  }

  std::vector<int> src_poison_states =
      GetPoisonedState(src_buffer_beg, src_buffer_end);
  std::vector<int> dst_poison_before =
      GetPoisonedState(dst_buffer_beg, dst_buffer_end);
  __sanitizer_copy_contiguous_container_annotations(src_beg, src_end, dst_beg,
                                                    dst_end);
  std::vector<int> dst_poison_after =
      GetPoisonedState(dst_buffer_beg, dst_buffer_end);

  // Create ideal copy of src over dst.
  std::vector<int> dst_poison_exp = dst_poison_before;
  for (size_t cur = 0; cur < capacity; ++cur)
    dst_poison_exp[off_dst + cur] = src_poison_states[off_src + cur];

  // Unpoison prefixes of Asan granules.
  for (size_t cur = dst_buffer_size - 1; cur > 0; --cur) {
    if (cur % kGranularity != 0 && !dst_poison_exp[cur])
      dst_poison_exp[cur - 1] = 0;
  }

  if (dst_poison_after != dst_poison_exp) {
    std::cerr << "[" << off_dst << ", " << off_dst + capacity << ")\n";
    for (size_t i = 0; i < dst_poison_after.size(); ++i) {
      std::cerr << i << ":\t" << dst_poison_before[i] << "\t"
                << dst_poison_after[i] << "\t" << dst_poison_exp[i] << "\n";
    }
    std::cerr << "----------\n";

    assert(dst_poison_after == dst_poison_exp);
  }
}

template <bool benchmark>
static void TestNonOverlappingContainers(size_t capacity, size_t off_src,
                                         size_t off_dst) {
  // Test will copy [off_src, off_src + capacity) to [off_dst, off_dst + capacity).
  // Allocate buffers to have additional granule before and after tested ranges.
  off_src += kGranularity;
  off_dst += kGranularity;
  size_t src_buffer_size = RoundUp(off_src + capacity) + kGranularity;
  size_t dst_buffer_size = RoundUp(off_dst + capacity) + kGranularity;

  std::unique_ptr<char[]> src_buffer =
      std::make_unique<char[]>(src_buffer_size);
  std::unique_ptr<char[]> dst_buffer =
      std::make_unique<char[]>(dst_buffer_size);

  char *src_buffer_beg = src_buffer.get();
  char *src_buffer_end = src_buffer_beg + src_buffer_size;
  assert(RoundDown(src_buffer_beg) == src_buffer_beg);

  char *dst_buffer_beg = dst_buffer.get();
  char *dst_buffer_end = dst_buffer_beg + dst_buffer_size;
  assert(RoundDown(dst_buffer_beg) == dst_buffer_beg);

  for (int i = 0; i < 35; i++) {
    if (!benchmark || !i) {
      RandomPoison(src_buffer_beg, src_buffer_end);
      RandomPoison(dst_buffer_beg, dst_buffer_end);
    }

    Test<benchmark>(capacity, off_src, off_dst, src_buffer_beg, src_buffer_end,
                    dst_buffer_beg, dst_buffer_end);
  }

  __asan_unpoison_memory_region(src_buffer_beg, src_buffer_size);
  __asan_unpoison_memory_region(dst_buffer_beg, dst_buffer_size);
}

template <bool benchmark>
static void TestOverlappingContainers(size_t capacity, size_t off_src,
                                      size_t off_dst) {
  // Test will copy [off_src, off_src + capacity) to [off_dst, off_dst + capacity).
  // Allocate buffers to have additional granule before and after tested ranges.
  off_src += kGranularity;
  off_dst += kGranularity;
  size_t buffer_size =
      RoundUp(std::max(off_src, off_dst) + capacity) + kGranularity;

  // Use unique_ptr with a custom deleter to manage the buffer
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);

  char *buffer_beg = buffer.get();
  char *buffer_end = buffer_beg + buffer_size;
  assert(RoundDown(buffer_beg) == buffer_beg);

  for (int i = 0; i < 35; i++) {
    if (!benchmark || !i)
      RandomPoison(buffer_beg, buffer_end);
    Test<benchmark>(capacity, off_src, off_dst, buffer_beg, buffer_end,
                    buffer_beg, buffer_end);
  }

  __asan_unpoison_memory_region(buffer_beg, buffer_size);
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (size_t off_src = 0; off_src < kGranularity; off_src++) {
    for (size_t off_dst = 0; off_dst < kGranularity; off_dst++) {
      for (int capacity = 0; capacity <= n; capacity++) {
        if (n < 1024) {
          TestNonOverlappingContainers<false>(capacity, off_src, off_dst);
          TestOverlappingContainers<false>(capacity, off_src, off_dst);
        } else {
          TestNonOverlappingContainers<true>(capacity, off_src, off_dst);
          TestOverlappingContainers<true>(capacity, off_src, off_dst);
        }
      }
    }
  }
}