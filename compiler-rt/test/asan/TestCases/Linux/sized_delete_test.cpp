// RUN: %clangxx_asan -fsized-deallocation -O0 %s -o %t
// RUN:                                                                  not %run %t scalar 2>&1 | FileCheck %s -check-prefix=SCALAR
// RUN: %env_asan_opts=new_delete_type_mismatch=1 not %run %t scalar 2>&1 | FileCheck %s -check-prefix=SCALAR
// RUN:                                                                  not %run %t array  2>&1 | FileCheck %s -check-prefix=ARRAY
// RUN: %env_asan_opts=new_delete_type_mismatch=1 not %run %t array  2>&1 | FileCheck %s -check-prefix=ARRAY
// RUN: %env_asan_opts=new_delete_type_mismatch=0     %run %t scalar
// RUN: %env_asan_opts=new_delete_type_mismatch=0     %run %t array

#include <new>
#include <stdio.h>
#include <string>

inline void break_optimization(void *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
}

struct S12 {
  int a, b, c;
};

struct S20 {
  int a, b, c, d, e;
};

struct D1 {
  int a, b, c;
  ~D1() { fprintf(stderr, "D1::~D1\n"); }
};

struct D2 {
  int a, b, c, d, e;
  ~D2() { fprintf(stderr, "D2::~D2\n"); }
};

void Del12(S12 *x) {
  break_optimization(x);
  delete x;
}
void Del12NoThrow(S12 *x) {
  break_optimization(x);
  operator delete(x, std::nothrow);
}
void Del12Ar(S12 *x) {
  break_optimization(x);
  delete [] x;
}
void Del12ArNoThrow(S12 *x) {
  break_optimization(x);
  operator delete[](x, std::nothrow);
}

int main(int argc, char **argv) {
  if (argc != 2) return 1;
  std::string flag = argv[1];
  // These are correct.
  Del12(new S12);
  Del12NoThrow(new S12);
  Del12Ar(new S12[100]);
  Del12ArNoThrow(new S12[100]);

  // Here we pass wrong type of pointer to delete,
  // but [] and nothrow variants of delete are not sized.
  Del12Ar(reinterpret_cast<S12*>(new S20[100]));
  Del12NoThrow(reinterpret_cast<S12*>(new S20));
  Del12ArNoThrow(reinterpret_cast<S12*>(new S20[100]));
  fprintf(stderr, "OK SO FAR\n");
  // SCALAR: OK SO FAR
  // ARRAY: OK SO FAR
  if (flag == "scalar") {
    // Here asan should bark as we are passing a wrong type of pointer
    // to sized delete.
    Del12(reinterpret_cast<S12*>(new S20));
    // SCALAR: AddressSanitizer: new-delete-type-mismatch
    // SCALAR:  object passed to delete has wrong type:
    // SCALAR:  size of the allocated type:   20 bytes;
    // SCALAR:  size of the deallocated type: 12 bytes.
    // SCALAR: is located 0 bytes inside of 20-byte region
    // SCALAR: SUMMARY: AddressSanitizer: new-delete-type-mismatch
  } else if (flag == "array") {
    D1 *d1 = reinterpret_cast<D1*>(new D2[10]);
    break_optimization(d1);
    delete [] d1;
    // ARRAY-NOT: D2::~D2
    // ARRAY: D1::~D1
    // ARRAY: AddressSanitizer: new-delete-type-mismatch
    // ARRAY:  size of the allocated type:   20{{4|8}} bytes;
    // ARRAY:  size of the deallocated type: 12{{4|8}} bytes.
  }
}
