// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_annotate_contiguous_container.

#include <algorithm>
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

static std::vector<int> GetPoisonedState(char *begin, char *end) {
  std::vector<int> result;
  for (; begin != end;) {
    int poisoned = 0;
    for (; begin != end && __asan_address_is_poisoned(begin); ++begin)
      ++poisoned;
    result.push_back(poisoned);
    int unpoisoned = 0;
    for (; begin != end && !__asan_address_is_poisoned(begin); ++begin)
      ++unpoisoned;
    result.push_back(unpoisoned);
  }
  return result;
}

static int GetFirstMismatch(const std::vector<int> &a,
                            const std::vector<int> &b) {
  auto mismatch = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
  return std::accumulate(a.begin(), mismatch.first, 0) +
         std::min(*mismatch.first, *mismatch.second);
}

void TestContainer(size_t capacity, size_t off_begin, bool poison_buffer) {
  size_t buffer_size = capacity + off_begin + kGranularity * 2;
  char *buffer = new char[buffer_size];
  if (poison_buffer)
    __asan_poison_memory_region(buffer, buffer_size);
  char *st_beg = buffer + off_begin;
  char *st_end = st_beg + capacity;
  char *end = poison_buffer ? st_beg : st_end;

  for (int i = 0; i < 1000; i++) {
    size_t size = rand() % (capacity + 1);
    assert(size <= capacity);
    char *old_end = end;
    end = st_beg + size;
    __sanitizer_annotate_contiguous_container(st_beg, st_end, old_end, end);

    char *cur = buffer;
    for (; cur < RoundDown(st_beg); ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
    // The prefix of the first incomplete granule can switch from poisoned to
    // unpoisoned but not otherwise.
    for (; cur < st_beg; ++cur)
      assert(poison_buffer || !__asan_address_is_poisoned(cur));
    for (; cur < end; ++cur)
      assert(!__asan_address_is_poisoned(cur));
    for (; cur < RoundDown(st_end); ++cur)
      assert(__asan_address_is_poisoned(cur));
    // The suffix of the last incomplete granule must be poisoned the same as
    // bytes after the end.
    for (; cur != st_end + kGranularity; ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
  }

  // Precalculate masks.
  std::vector<std::vector<int>> masks(capacity + 1);
  for (int i = 0; i <= capacity; i++) {
    char *old_end = end;
    end = st_beg + i;
    __sanitizer_annotate_contiguous_container(st_beg, st_end, old_end, end);
    masks[i] = GetPoisonedState(st_beg, st_end);
  }
  for (int i = 0; i <= capacity; i++) {
    char *old_end = end;
    end = st_beg + i;
    __sanitizer_annotate_contiguous_container(st_beg, st_end, old_end, end);

    char *cur_first = std::max(end - 2 * kGranularity, st_beg);
    char *cur_last = std::min(end + 2 * kGranularity, st_end);
    for (char *cur = cur_first; cur <= cur_last; ++cur) {
      bool is_valid =
          __sanitizer_verify_contiguous_container(st_beg, cur, st_end);
      const void *bad_address =
          __sanitizer_contiguous_container_find_bad_address(st_beg, cur,
                                                            st_end);
      if (cur == end ||
          // The last unaligned granule of the storage followed by unpoisoned
          // bytes looks the same.
          (!poison_buffer && RoundDown(st_end) <= std::min(cur, end))) {
        assert(is_valid);
        assert(!bad_address);
        continue;
      }
      assert(!is_valid);
      assert(bad_address == std::min(cur, end));
      assert(bad_address ==
             st_beg + GetFirstMismatch(masks[i], masks[cur - st_beg]));
    }
  }

  __asan_unpoison_memory_region(buffer, buffer_size);
  delete[] buffer;
}

void TestDoubleEndedContainer(size_t capacity, size_t off_begin,
                              bool poison_buffer) {
  size_t buffer_size = capacity + off_begin + kGranularity * 2;
  char *buffer = new char[buffer_size];
  if (poison_buffer)
    __asan_poison_memory_region(buffer, buffer_size);
  char *st_beg = buffer + off_begin;
  char *st_end = st_beg + capacity;
  char *beg = st_beg;
  char *end = poison_buffer ? st_beg : st_end;

  for (int i = 0; i < 1000; i++) {
    size_t size = rand() % (capacity + 1);
    size_t skipped = rand() % (capacity - size + 1);
    assert(size <= capacity);
    char *old_beg = beg;
    char *old_end = end;
    beg = st_beg + skipped;
    end = beg + size;

    __sanitizer_annotate_double_ended_contiguous_container(
        st_beg, st_end, old_beg, old_end, beg, end);

    char *cur = buffer;
    for (; cur < RoundDown(st_beg); ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
    // The prefix of the first incomplete granule can switch from poisoned to
    // unpoisoned but not otherwise.
    for (; cur < st_beg; ++cur)
      assert(poison_buffer || !__asan_address_is_poisoned(cur));
    if (beg != end) {
      for (; cur < RoundDown(beg); ++cur)
        assert(__asan_address_is_poisoned(cur));

      for (; cur < end; ++cur)
        assert(!__asan_address_is_poisoned(cur));
    }
    for (; cur < RoundDown(st_end); ++cur)
      assert(__asan_address_is_poisoned(cur));
    // The suffix of the last incomplete granule must be poisoned the same as
    // bytes after the end.
    for (; cur != st_end + kGranularity; ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
  }

  if (capacity < 32) {

    // Precalculate masks.
    std::vector<std::vector<std::vector<int>>> masks(
        capacity + 1, std::vector<std::vector<int>>(capacity + 1));
    for (int i = 0; i <= capacity; i++) {
      for (int j = i; j <= capacity; j++) {
        char *old_beg = beg;
        char *old_end = end;
        beg = st_beg + i;
        end = st_beg + j;
        __sanitizer_annotate_double_ended_contiguous_container(
            st_beg, st_end, old_beg, old_end, beg, end);
        masks[i][j] = GetPoisonedState(st_beg, st_end);
      }
    }

    for (int i = 0; i <= capacity; i++) {
      for (int j = i; j <= capacity; j++) {
        char *old_beg = beg;
        char *old_end = end;
        beg = st_beg + i;
        end = st_beg + j;
        __sanitizer_annotate_double_ended_contiguous_container(
            st_beg, st_end, old_beg, old_end, beg, end);

        // Try to mismatch the end of the container.
        char *cur_first = std::max(end - 2 * kGranularity, beg);
        char *cur_last = std::min(end + 2 * kGranularity, st_end);
        for (char *cur = cur_first; cur <= cur_last; ++cur) {
          bool is_valid = __sanitizer_verify_double_ended_contiguous_container(
              st_beg, beg, cur, st_end);
          const void *bad_address =
              __sanitizer_double_ended_contiguous_container_find_bad_address(
                  st_beg, beg, cur, st_end);

          if (cur == end ||
              // The last unaligned granule of the storage followed by unpoisoned
              // bytes looks the same.
              (!poison_buffer && RoundDown(st_end) <= std::min(cur, end))) {
            assert(is_valid);
            assert(!bad_address);
            continue;
          }

          assert(!is_valid);
          assert(bad_address);
          assert(bad_address ==
                 st_beg +
                     GetFirstMismatch(masks[i][j], masks[i][cur - st_beg]));
        }

        // Try to mismatch the begin of the container.
        cur_first = std::max(beg - 2 * kGranularity, st_beg);
        cur_last = std::min(beg + 2 * kGranularity, end);
        for (char *cur = cur_first; cur <= cur_last; ++cur) {
          bool is_valid = __sanitizer_verify_double_ended_contiguous_container(
              st_beg, cur, end, st_end);
          const void *bad_address =
              __sanitizer_double_ended_contiguous_container_find_bad_address(
                  st_beg, cur, end, st_end);

          if (cur == beg ||
              // The last unaligned granule of the storage followed by unpoisoned
              // bytes looks the same.
              (!poison_buffer && RoundDown(st_end) <= std::min(cur, beg) ||
               // The first unaligned granule of non-empty container looks the
               // same.
               (std::max(beg, cur) < end &&
                RoundDown(beg) == RoundDown(cur)))) {
            assert(is_valid);
            assert(!bad_address);
            continue;
          }
          assert(!is_valid);
          assert(bad_address);
          assert(bad_address ==
                 st_beg +
                     GetFirstMismatch(masks[i][j], masks[cur - st_beg][j]));
        }
      }
    }
  }

  __asan_unpoison_memory_region(buffer, buffer_size);
  delete[] buffer;
}

__attribute__((noinline)) void Throw() { throw 1; }

__attribute__((noinline)) void ThrowAndCatch() {
  try {
    Throw();
  } catch (...) {
  }
}

void TestThrow() {
  char x[32];
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 32, x + 14);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(__asan_address_is_poisoned(x + 14));
  ThrowAndCatch();
  assert(!__asan_address_is_poisoned(x + 13));
  assert(!__asan_address_is_poisoned(x + 14));
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 14, x + 32);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(!__asan_address_is_poisoned(x + 14));
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j < kGranularity * 2; j++) {
      for (int poison = 0; poison < 2; ++poison) {
        TestContainer(i, j, poison);
        TestDoubleEndedContainer(i, j, poison);
      }
    }
  }
  TestThrow();
}
