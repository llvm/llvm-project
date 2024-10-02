// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_copy_contiguous_container_annotations.

#include <algorithm>
#include <deque>
#include <memory>
#include <numeric>

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

static std::deque<int> GetPoisonedState(char *begin, char *end) {
  std::deque<int> result;
  for (char *ptr = begin; ptr != end; ++ptr) {
    result.push_back(__asan_address_is_poisoned(ptr));
  }
  return result;
}

static void RandomPoison(char *beg, char *end) {
  if (beg != RoundDown(beg) && RoundDown(beg) != RoundDown(end) &&
      rand() % 2 == 1) {
    __asan_poison_memory_region(beg, RoundUp(beg) - beg);
    __asan_unpoison_memory_region(beg, rand() % (RoundUp(beg) - beg + 1));
  }
  for (beg = RoundUp(beg); beg + kGranularity <= end; beg += kGranularity) {
    __asan_poison_memory_region(beg, kGranularity);
    __asan_unpoison_memory_region(beg, rand() % (kGranularity + 1));
  }
  if (end > beg && __asan_address_is_poisoned(end)) {
    __asan_poison_memory_region(beg, kGranularity);
    __asan_unpoison_memory_region(beg, rand() % (end - beg + 1));
  }
}

static size_t CountUnpoisoned(std::deque<int> &poison_states, size_t n) {
  size_t result = 0;
  for (size_t i = 0; i < n && !poison_states.empty(); ++i) {
    if (!poison_states.front()) {
      result = i + 1;
    }
    poison_states.pop_front();
  }

  return result;
}

void TestNonOverlappingContainers(size_t capacity, size_t off_src,
                                  size_t off_dst, bool poison_src,
                                  bool poison_dst) {
  size_t src_buffer_size = capacity + off_src + kGranularity * 2;
  size_t dst_buffer_size = capacity + off_dst + kGranularity * 2;

  std::unique_ptr<char[]> src_buffer =
      std::make_unique<char[]>(src_buffer_size);
  std::unique_ptr<char[]> dst_buffer =
      std::make_unique<char[]>(dst_buffer_size);

  char *src_buffer_end = src_buffer.get() + src_buffer_size;
  char *dst_buffer_end = dst_buffer.get() + dst_buffer_size;
  char *src_beg = src_buffer.get() + off_src;
  char *dst_beg = dst_buffer.get() + off_dst;
  char *src_end = src_beg + capacity;
  char *dst_end = dst_beg + capacity;

  for (int i = 0; i < 35; i++) {
    if (poison_src)
      __asan_poison_memory_region(src_buffer.get(), src_buffer_size);
    if (poison_dst)
      __asan_poison_memory_region(dst_buffer.get(), dst_buffer_size);

    RandomPoison(src_beg, src_end);
    std::deque<int> poison_states = GetPoisonedState(src_beg, src_end);
    __sanitizer_copy_contiguous_container_annotations(src_beg, src_end, dst_beg,
                                                      dst_end);

    // If src_buffer were poisoned, expected state of memory before src_beg
    // is undetermined.
    // If old buffer were not poisoned, that memory should still be unpoisoned.
    char *cur;
    if (!poison_src) {
      for (cur = src_buffer.get(); cur < src_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }
    for (size_t i = 0; i < poison_states.size(); ++i) {
      assert(__asan_address_is_poisoned(&src_beg[i]) == poison_states[i]);
    }
    // Memory after src_end should be the same as at the beginning.
    for (cur = src_end; cur < src_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_dst);
    }

    // If dst_buffer were not poisoned, memory before dst_beg should never
    // be poisoned. Otherwise, its state is undetermined.
    if (!poison_dst) {
      for (cur = dst_buffer.get(); cur < dst_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }

    char *next;
    for (cur = dst_beg; cur + kGranularity <= dst_end; cur = next) {
      next = RoundUp(cur + 1);
      size_t unpoisoned = CountUnpoisoned(poison_states, next - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < next) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // [cur; dst_end) is not checked yet.
    // If dst_buffer were not poisoned, it cannot be poisoned.
    // If dst_buffer were poisoned, it should be same as earlier.
    if (cur < dst_end) {
      size_t unpoisoned = CountUnpoisoned(poison_states, dst_end - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < dst_end && poison_dst) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // Memory annotations after dst_end should be unchanged.
    for (cur = dst_end; cur < dst_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_dst);
    }
  }

  __asan_unpoison_memory_region(src_buffer.get(), src_buffer_size);
  __asan_unpoison_memory_region(dst_buffer.get(), dst_buffer_size);
}

void TestOverlappingContainers(size_t capacity, size_t off_src, size_t off_dst,
                               bool poison_whole, bool poison_dst) {
  size_t buffer_size = capacity + off_src + off_dst + kGranularity * 3;

  // Use unique_ptr with a custom deleter to manage the buffer
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);

  char *buffer_end = buffer.get() + buffer_size;
  char *src_beg = buffer.get() + kGranularity + off_src;
  char *dst_beg = buffer.get() + kGranularity + off_dst;
  char *src_end = src_beg + capacity;
  char *dst_end = dst_beg + capacity;

  for (int i = 0; i < 35; i++) {
    if (poison_whole)
      __asan_poison_memory_region(buffer.get(), buffer_size);
    if (poison_dst)
      __asan_poison_memory_region(dst_beg, dst_end - dst_beg);

    RandomPoison(src_beg, src_end);
    auto poison_states = GetPoisonedState(src_beg, src_end);
    __sanitizer_copy_contiguous_container_annotations(src_beg, src_end, dst_beg,
                                                      dst_end);
    // This variable is used only when buffer ends in the middle of a granule.
    bool can_modify_last_granule = __asan_address_is_poisoned(dst_end);

    // If whole buffer were poisoned, expected state of memory before first container
    // is undetermined.
    // If old buffer were not poisoned, that memory should still be unpoisoned.
    char *cur;
    if (!poison_whole) {
      for (cur = buffer.get(); cur < src_beg && cur < dst_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }

    // Memory after end of both containers should be the same as at the beginning.
    for (cur = (src_end > dst_end) ? src_end : dst_end; cur < buffer_end;
         ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_whole);
    }

    char *next;
    for (cur = dst_beg; cur + kGranularity <= dst_end; cur = next) {
      next = RoundUp(cur + 1);
      size_t unpoisoned = CountUnpoisoned(poison_states, next - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < next) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // [cur; dst_end) is not checked yet, if container ends in the middle of a granule.
    // It can be poisoned, only if non-container bytes in that granule were poisoned.
    // Otherwise, it should be unpoisoned.
    if (cur < dst_end) {
      size_t unpoisoned = CountUnpoisoned(poison_states, dst_end - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < dst_end && can_modify_last_granule) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
  }

  __asan_unpoison_memory_region(buffer.get(), buffer_size);
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (size_t off_src = 0; off_src < kGranularity + 2; off_src++) {
    for (size_t off_dst = 0; off_dst < kGranularity + 2; off_dst++) {
      for (int capacity = 0; capacity <= n; capacity++) {
        for (int poison_dst = 0; poison_dst < 2; ++poison_dst) {
          for (int poison_dst = 0; poison_dst < 2; ++poison_dst) {
            TestNonOverlappingContainers(capacity, off_src, off_dst, poison_dst,
                                         poison_dst);
            TestOverlappingContainers(capacity, off_src, off_dst, poison_dst,
                                      poison_dst);
          }
        }
      }
    }
  }
}
