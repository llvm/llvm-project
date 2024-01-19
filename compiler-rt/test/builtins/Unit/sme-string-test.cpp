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

  Memory(unsigned Stride = 0) {
    size = N;
    if (Stride == 0)
      return;
    for (unsigned I = 0, Elem = 0; I < N; I++) {
      ptr[I] = I * Stride;
    }
  }

  void assert_equal(const Memory &Other) {
    assert(N == Other.size);
    assert(memcmp(ptr, Other.ptr, N) == 0);
  }

  void assert_equal(std::initializer_list<uint8_t> S) {
    assert(N == S.size());
    auto It = S.begin();
    for (unsigned I = 0; I < N; I++) {
      assert(ptr[I] == *It++);
    }
  }

  void assert_elemt_equal_at(unsigned I, uint8_t elem) {
    assert(ptr[I] == elem);
  }
};

int main() {

  // Testing memcpy from Src to Dst.
  {
    Memory<8> Src(1);
    Memory<8> Dst;
    if (!__arm_sc_memcpy(Dst.ptr, Src.ptr, 8))
      abort();
    Dst.assert_equal(Src);
    Dst.assert_equal({0, 1, 2, 3, 4, 5, 6, 7});
  }

  // Testing memcpy from Src to Dst with pointer offset.
  {
    Memory<8> Src(1);
    Memory<8> Dst(1);
    if (!__arm_sc_memcpy(Dst.ptr + 1, Src.ptr, 6))
      abort();
    Dst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
  }

  // Testing memchr.
  {
    Memory<8> Src(4);
    for (unsigned I = 0; I < 8; I++) {
      uint8_t E = Src.ptr[I];
      uint8_t *Elem = (uint8_t *)memchr(Src.ptr, E, 8);
      if (!Elem)
        abort();
      Src.assert_elemt_equal_at(Elem - Src.ptr, *Elem);
      for (unsigned I = 0; I < 8; ++I)
        assert(__arm_sc_memchr(Src.ptr, Src.ptr[I], 8) ==
               memchr(Src.ptr, Src.ptr[I], 8));
    }
  }

  // Testing memset.
  {
    Memory<8> Array;
    if (!__arm_sc_memset(Array.ptr, 2, 8))
      abort();
    Array.assert_equal({2, 2, 2, 2, 2, 2, 2, 2});
  }

  // Testing memset with pointer offset.
  {
    Memory<8> Array(1);
    if (!__arm_sc_memset(Array.ptr + 1, 2, 6))
      abort();
    Array.assert_equal({0, 2, 2, 2, 2, 2, 2, 7});
  }

  // Testing memmove with a simple non-overlap case.
  {
    Memory<8> Src(1);
    Memory<8> Dst(1);
    if (!__arm_sc_memmove(Dst.ptr + 1, Src.ptr, 6))
      abort();
    Dst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
  }

  // Testing memove with overlap pointers Dst > Src, Dst < Src.
  {
    Memory<8> SrcDst(1);
    if (!__arm_sc_memmove(SrcDst.ptr + 1, SrcDst.ptr, 6))
      abort();
    SrcDst.assert_equal({0, 0, 1, 2, 3, 4, 5, 7});
    if (!__arm_sc_memmove(SrcDst.ptr, SrcDst.ptr + 1, 6))
      abort();
    SrcDst.assert_equal({0, 1, 2, 3, 4, 5, 5, 7});
  }

  return 0;
}
