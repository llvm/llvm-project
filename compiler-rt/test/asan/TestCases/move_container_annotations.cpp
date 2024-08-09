// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_move_contiguous_container_annotations.

#include <algorithm>
#include <deque>
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
  return (x == RoundDown(x))
             ? x
             : reinterpret_cast<T>(reinterpret_cast<uintptr_t>(RoundDown(x)) +
                                   kGranularity);
}

static std::deque<int> GetPoisonedState(char *begin, char *end) {
  std::deque<int> result;
  for (; begin != end; ++begin) {
    result.push_back(__asan_address_is_poisoned(begin));
  }
  return result;
}

static void RandomPoison(char *beg, char *end) {
  if (beg != RoundDown(beg) && (rand() % 2 == 1)) {
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

static size_t count_unpoisoned(std::deque<int> &poison_states, size_t n) {
  size_t result = 0;
  for (size_t i = 0; i < n && !poison_states.empty(); ++i) {
    if (!poison_states.front()) {
      result = i + 1;
    }
    poison_states.pop_front();
  }

  return result;
}

void TestMove(size_t capacity, size_t off_old, size_t off_new,
              int poison_buffers) {
  size_t old_buffer_size = capacity + off_old + kGranularity * 2;
  size_t new_buffer_size = capacity + off_new + kGranularity * 2;
  char *old_buffer = new char[old_buffer_size];
  char *new_buffer = new char[new_buffer_size];
  char *old_buffer_end = old_buffer + old_buffer_size;
  char *new_buffer_end = new_buffer + new_buffer_size;
  bool poison_old = poison_buffers % 2 == 1;
  bool poison_new = poison_buffers / 2 == 1;
  if (poison_old)
    __asan_poison_memory_region(old_buffer, old_buffer_size);
  if (poison_new)
    __asan_poison_memory_region(new_buffer, new_buffer_size);
  char *old_beg = old_buffer + off_old;
  char *new_beg = new_buffer + off_new;
  char *old_end = old_beg + capacity;
  char *new_end = new_beg + capacity;

  for (int i = 0; i < 1000; i++) {
    RandomPoison(old_beg, old_end);
    std::deque<int> poison_states(old_beg, old_end);
    __sanitizer_move_contiguous_container_annotations(old_beg, old_end, new_beg,
                                                      new_end);

    // If old_buffer were poisoned, expected state of memory before old_beg
    // is undetermined.
    // If old buffer were not poisoned, that memory should still be unpoisoned.
    // Area between old_beg and old_end should never be poisoned.
    char *cur = poison_old ? old_beg : old_buffer;
    for (; cur < old_end; ++cur) {
      assert(!__asan_address_is_poisoned(cur));
    }
    // Memory after old_beg should be the same as at the beginning.
    for (; cur < old_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_old);
    }

    // If new_buffer were not poisoned, memory before new_beg should never
    // be poisoned. Otherwise, its state is undetermined.
    if (!poison_new) {
      for (cur = new_buffer; cur < new_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }
    //In every granule, poisoned memory should be after last expected unpoisoned.
    char *next;
    for (cur = new_beg; cur + kGranularity <= new_end; cur = next) {
      next = RoundUp(cur + 1);
      size_t unpoisoned = count_unpoisoned(poison_states, next - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < next) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // [cur; new_end) is not checked yet.
    // If new_buffer were not poisoned, it cannot be poisoned and we can ignore check.
    // If new_buffer were poisoned, it should be same as earlier.
    if (cur < new_end && poison_new) {
      size_t unpoisoned = count_unpoisoned(poison_states, new_end - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < new_end) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // Memory annotations after new_end should be unchanged.
    for (cur = new_end; cur < new_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_new);
    }
  }

  __asan_unpoison_memory_region(old_buffer, old_buffer_size);
  __asan_unpoison_memory_region(new_buffer, new_buffer_size);
  delete[] old_buffer;
  delete[] new_buffer;
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j < kGranularity * 2; j++) {
      for (int k = 0; k < kGranularity * 2; k++) {
        for (int poison = 0; poison < 4; ++poison) {
          TestMove(i, j, k, poison);
        }
      }
    }
  }
}
