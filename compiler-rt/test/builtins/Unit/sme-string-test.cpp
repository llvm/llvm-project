// REQUIRES: aarch64-target-arch, aarch64-sme-available
// RUN: %clangxx_builtins %s %librt -o %t && %run %t

#include <cassert>
#include <initializer_list>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
void *__arm_sc_memcpy(void *, const void *, size_t);
void *__arm_sc_memset(void *, int, size_t);
void *__arm_sc_memmove(void *, const void *, size_t);
void *__arm_sc_memchr(const void *, int, size_t);
}

template <unsigned N> class Memory {
public:
  uint8_t ptr[N];
  unsigned size;

  Memory(unsigned stride = 0) {
    size = N;
    if (stride == 0)
      return;
    for (unsigned i = 0; i < N; i++)
      ptr[i] = i * stride;
  }

  void assert_equal(const Memory &other) {
    assert(N == other.size);
    assert(memcmp(ptr, other.ptr, N) == 0);
  }

  void assert_equal(std::initializer_list<uint8_t> s) {
    assert(N == s.size());
    auto it = s.begin();
    for (unsigned i = 0; i < N; ++i)
      assert(ptr[i] == *it++);
  }

  void assert_elemt_equal_at(unsigned I, uint8_t elem) {
    assert(ptr[I] == elem);
  }
};

int main() {

  // Testing memcpy from src to dst.
  {
    Memory<8> src(1);
    Memory<8> dst;
    if (!__arm_sc_memcpy(dst.ptr, src.ptr, 8))
      abort();
    dst.assert_equal(src);
    dst.assert_equal({0, 1, 2, 3, 4, 5, 6, 7});
  }

  // Testing memcpy from src to dst with pointer offset.
  {
    Memory<8> src(1);
    Memory<8> dst(1);
    if (!__arm_sc_memcpy(dst.ptr + 1, src.ptr, 6))
      abort();
    dst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
  }

  // Testing memchr.
  {
    Memory<8> src(4);
    for (unsigned i = 0; i < 8; ++i) {
      uint8_t e = src.ptr[i];
      uint8_t *elem = (uint8_t *)memchr(src.ptr, e, 8);
      if (!elem)
        abort();
      src.assert_elemt_equal_at(elem - src.ptr, *elem);
      for (unsigned i = 0; i < 8; ++i)
        assert(__arm_sc_memchr(src.ptr, src.ptr[i], 8) ==
               memchr(src.ptr, src.ptr[i], 8));
    }
  }

  // Testing memset.
  {
    Memory<8> array;
    if (!__arm_sc_memset(array.ptr, 2, 8))
      abort();
    array.assert_equal({2, 2, 2, 2, 2, 2, 2, 2});
  }

  // Testing memset with pointer offset.
  {
    Memory<8> array(1);
    if (!__arm_sc_memset(array.ptr + 1, 2, 6))
      abort();
    array.assert_equal({0, 2, 2, 2, 2, 2, 2, 7});
  }

  // Testing memmove with a simple non-overlap case.
  {
    Memory<8> src(1);
    Memory<8> dst(1);
    if (!__arm_sc_memmove(dst.ptr + 1, src.ptr, 6))
      abort();
    dst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
  }

  // Testing memove with overlap pointers dst > src, dst < src.
  {
    Memory<8> srcdst(1);
    if (!__arm_sc_memmove(srcdst.ptr + 1, srcdst.ptr, 6))
      abort();
    srcdst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
    if (!__arm_sc_memmove(srcdst.ptr, srcdst.ptr + 1, 6))
      abort();
    srcdst.assert_equal({0, 1, 2, 3, 4, 5, 5, 7});
  }

  return 0;
}
