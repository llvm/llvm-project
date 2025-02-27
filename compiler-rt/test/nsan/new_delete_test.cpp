/// From sanitizer_common/TestCases/Linux/new_delete_test.cpp
// RUN: %clangxx_nsan -fno-sized-deallocation -O0 %s -o %t && %run %t
// RUN: %clangxx_nsan -fsized-deallocation -O0 %s -o %t && %run %t

#include <cstddef>

namespace std {
struct nothrow_t {};
static const nothrow_t nothrow;
enum class align_val_t : size_t {};
} // namespace std

void *operator new(size_t);
void *operator new[](size_t);
void *operator new(size_t, std::nothrow_t const &);
void *operator new[](size_t, std::nothrow_t const &);
void *operator new(size_t, std::align_val_t);
void *operator new[](size_t, std::align_val_t);
void *operator new(size_t, std::align_val_t, std::nothrow_t const &);
void *operator new[](size_t, std::align_val_t, std::nothrow_t const &);

void operator delete(void *) throw();
void operator delete[](void *) throw();
void operator delete(void *, std::nothrow_t const &);
void operator delete[](void *, std::nothrow_t const &);
void operator delete(void *, size_t) throw();
void operator delete[](void *, size_t) throw();
void operator delete(void *, std::align_val_t) throw();
void operator delete[](void *, std::align_val_t) throw();
void operator delete(void *, std::align_val_t, std::nothrow_t const &);
void operator delete[](void *, std::align_val_t, std::nothrow_t const &);
void operator delete(void *, size_t, std::align_val_t) throw();
void operator delete[](void *, size_t, std::align_val_t) throw();

template <typename T> inline T *break_optimization(T *arg) {
  __asm__ __volatile__("" : : "r"(arg) : "memory");
  return arg;
}

struct S12 {
  int a, b, c;
};
struct alignas(128) S12_128 {
  int a, b, c;
};
struct alignas(256) S12_256 {
  int a, b, c;
};
struct alignas(512) S1024_512 {
  char a[1024];
};
struct alignas(1024) S1024_1024 {
  char a[1024];
};

int main(int argc, char **argv) {
  delete break_optimization(new S12);
  operator delete(break_optimization(new S12), std::nothrow);
  delete[] break_optimization(new S12[100]);
  operator delete[](break_optimization(new S12[100]), std::nothrow);

  delete break_optimization(new S12_128);
  operator delete(break_optimization(new S12_128),
                  std::align_val_t(alignof(S12_128)));
  operator delete(break_optimization(new S12_128),
                  std::align_val_t(alignof(S12_128)), std::nothrow);
  operator delete(break_optimization(new S12_128), sizeof(S12_128),
                  std::align_val_t(alignof(S12_128)));

  delete[] break_optimization(new S12_128[100]);
  operator delete[](break_optimization(new S12_128[100]),
                    std::align_val_t(alignof(S12_128)));
  operator delete[](break_optimization(new S12_128[100]),
                    std::align_val_t(alignof(S12_128)), std::nothrow);
  operator delete[](break_optimization(new S12_128[100]), sizeof(S12_128[100]),
                    std::align_val_t(alignof(S12_128)));
}
