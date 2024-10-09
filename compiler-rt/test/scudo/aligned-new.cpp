// RUN: %clangxx_scudo %s -o %t
// RUN:                                                 %run %t valid   2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=1     %run %t invalid 2>&1
// RUN: %env_scudo_opts=allocator_may_return_null=0 not %run %t invalid 2>&1 | FileCheck %s

// Tests that the C++17 aligned new/delete operators are working as expected.
// Currently we do not check the consistency of the alignment on deallocation,
// so this just tests that the APIs work.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Define all new/delete to not depend on the version provided by the platform.

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

template <typename T>
inline T *break_optimization(T *arg) {
  __asm__ __volatile__(""
                       :
                       : "r"(arg)
                       : "memory");
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
  assert(argc == 2);

  if (!strcmp(argv[1], "valid")) {
    // Standard use case.
    delete break_optimization(new S12);
    delete break_optimization(new S12_128);
    delete[] break_optimization(new S12_128[4]);
    delete break_optimization(new S12_256);
    delete break_optimization(new S1024_512);
    delete[] break_optimization(new S1024_512[4]);
    delete break_optimization(new S1024_1024);

    // Call directly the aligned versions of the operators.
    const size_t alignment = 1U << 8;
    void *p = operator new(1, static_cast<std::align_val_t>(alignment));
    assert((reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0);
    operator delete(p, static_cast<std::align_val_t>(alignment));
  }
  if (!strcmp(argv[1], "invalid")) {
    // Alignment must be a power of 2.
    const size_t alignment = (1U << 8) - 1;
    void *p = operator new(1, static_cast<std::align_val_t>(alignment),
                           std::nothrow);
    // CHECK: Scudo ERROR: invalid allocation alignment
    assert(!p);
  }

  return 0;
}
