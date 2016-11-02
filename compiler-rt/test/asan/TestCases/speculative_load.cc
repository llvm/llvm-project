// Verifies that speculative loads from unions do not happen under asan.
// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -O1 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -O2 %s -o %t && %run %t 2>&1
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1

#include <sanitizer/asan_interface.h>

struct S {
  struct _long {
      void* _pad;
      const char* _ptr;
  };

  struct _short {
    unsigned char _size;
    char _ch[23];
  };

  union {
    _short _s;
    _long _l;
  } _data;

  S() {
    _data._s._size = 0;
    __asan_poison_memory_region(_data._s._ch, 23);
  }

  ~S() {
    __asan_unpoison_memory_region(_data._s._ch, 23);
  }

  bool is_long() const {
    return _data._s._size & 1;
  }

  const char* get_pointer() const {
    return is_long() ? _data._l._ptr : _data._s._ch;
  }
};


inline void side_effect(const void *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
}

int main(int argc, char **argv) {
  S s;
  side_effect(&s); // optimizer is too smart otherwise
  const char *ptr = s.get_pointer();
  side_effect(ptr); // force use ptr
  return 0;
}
