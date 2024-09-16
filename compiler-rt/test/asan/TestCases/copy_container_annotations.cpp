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
  return (x == RoundDown(x))
             ? x
             : reinterpret_cast<T>(reinterpret_cast<uintptr_t>(RoundDown(x)) +
                                   kGranularity);
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

void TestNonOverlappingContainers(size_t capacity, size_t off_old,
                                  size_t off_new, int poison_buffers) {
  size_t old_buffer_size = capacity + off_old + kGranularity * 2;
  size_t new_buffer_size = capacity + off_new + kGranularity * 2;

  // Use unique_ptr with a custom deleter to manage the buffers
  std::unique_ptr<char[]> old_buffer =
      std::make_unique<char[]>(old_buffer_size);
  std::unique_ptr<char[]> new_buffer =
      std::make_unique<char[]>(new_buffer_size);

  char *old_buffer_end = old_buffer.get() + old_buffer_size;
  char *new_buffer_end = new_buffer.get() + new_buffer_size;
  bool poison_old = poison_buffers % 2 == 1;
  bool poison_new = poison_buffers / 2 == 1;
  char *old_beg = old_buffer.get() + off_old;
  char *new_beg = new_buffer.get() + off_new;
  char *old_end = old_beg + capacity;
  char *new_end = new_beg + capacity;

  for (int i = 0; i < 35; i++) {
    if (poison_old)
      __asan_poison_memory_region(old_buffer.get(), old_buffer_size);
    if (poison_new)
      __asan_poison_memory_region(new_buffer.get(), new_buffer_size);

    RandomPoison(old_beg, old_end);
    std::deque<int> poison_states = GetPoisonedState(old_beg, old_end);
    __sanitizer_copy_contiguous_container_annotations(old_beg, old_end, new_beg,
                                                      new_end);

    // If old_buffer were poisoned, expected state of memory before old_beg
    // is undetermined.
    // If old buffer were not poisoned, that memory should still be unpoisoned.
    char *cur;
    if (!poison_old) {
      for (cur = old_buffer.get(); cur < old_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }
    for (size_t i = 0; i < poison_states.size(); ++i) {
      assert(__asan_address_is_poisoned(&old_beg[i]) == poison_states[i]);
    }
    // Memory after old_end should be the same as at the beginning.
    for (cur = old_end; cur < old_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_old);
    }

    // If new_buffer were not poisoned, memory before new_beg should never
    // be poisoned. Otherwise, its state is undetermined.
    if (!poison_new) {
      for (cur = new_buffer.get(); cur < new_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }

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
    // If new_buffer were not poisoned, it cannot be poisoned.
    // If new_buffer were poisoned, it should be same as earlier.
    if (cur < new_end) {
      size_t unpoisoned = count_unpoisoned(poison_states, new_end - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < new_end && poison_new) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
    // Memory annotations after new_end should be unchanged.
    for (cur = new_end; cur < new_buffer_end; ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_new);
    }
  }

  __asan_unpoison_memory_region(old_buffer.get(), old_buffer_size);
  __asan_unpoison_memory_region(new_buffer.get(), new_buffer_size);
}

void TestOverlappingContainers(size_t capacity, size_t off_old, size_t off_new,
                               int poison_buffers) {
  size_t buffer_size = capacity + off_old + off_new + kGranularity * 3;

  // Use unique_ptr with a custom deleter to manage the buffer
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);

  char *buffer_end = buffer.get() + buffer_size;
  bool poison_whole = poison_buffers % 2 == 1;
  bool poison_new = poison_buffers / 2 == 1;
  char *old_beg = buffer.get() + kGranularity + off_old;
  char *new_beg = buffer.get() + kGranularity + off_new;
  char *old_end = old_beg + capacity;
  char *new_end = new_beg + capacity;

  for (int i = 0; i < 35; i++) {
    if (poison_whole)
      __asan_poison_memory_region(buffer.get(), buffer_size);
    if (poison_new)
      __asan_poison_memory_region(new_beg, new_end - new_beg);

    RandomPoison(old_beg, old_end);
    std::deque<int> poison_states = GetPoisonedState(old_beg, old_end);
    __sanitizer_copy_contiguous_container_annotations(old_beg, old_end, new_beg,
                                                      new_end);
    // This variable is used only when buffer ends in the middle of a granule.
    bool can_modify_last_granule = __asan_address_is_poisoned(new_end);

    // If whole buffer were poisoned, expected state of memory before first container
    // is undetermined.
    // If old buffer were not poisoned, that memory should still be unpoisoned.
    char *cur;
    if (!poison_whole) {
      for (cur = buffer.get(); cur < old_beg && cur < new_beg; ++cur) {
        assert(!__asan_address_is_poisoned(cur));
      }
    }

    // Memory after end of both containers should be the same as at the beginning.
    for (cur = (old_end > new_end) ? old_end : new_end; cur < buffer_end;
         ++cur) {
      assert(__asan_address_is_poisoned(cur) == poison_whole);
    }

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
    // [cur; new_end) is not checked yet, if container ends in the middle of a granule.
    // It can be poisoned, only if non-container bytes in that granule were poisoned.
    // Otherwise, it should be unpoisoned.
    if (cur < new_end) {
      size_t unpoisoned = count_unpoisoned(poison_states, new_end - cur);
      if (unpoisoned > 0) {
        assert(!__asan_address_is_poisoned(cur + unpoisoned - 1));
      }
      if (cur + unpoisoned < new_end && can_modify_last_granule) {
        assert(__asan_address_is_poisoned(cur + unpoisoned));
      }
    }
  }

  __asan_unpoison_memory_region(buffer.get(), buffer_size);
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (size_t j = 0; j < kGranularity + 2; j++) {
    for (size_t k = 0; k < kGranularity + 2; k++) {
      for (int i = 0; i <= n; i++) {
        for (int poison = 0; poison < 4; ++poison) {
          TestNonOverlappingContainers(i, j, k, poison);
          TestOverlappingContainers(i, j, k, poison);
        }
      }
    }
  }
}
