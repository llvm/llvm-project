// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -g %s -O3 -o %t
// RUN: %run %t rT && %run %t mT && %run %t fT && %run %t cT
// RUN: %run %t rU && %run %t mU && %run %t fU && %run %t cU
// RUN: %run %t rS && %run %t rV && %run %t oV
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t mS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --check-prefix=CHECK-%os-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t fS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t cS 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --check-prefix=CHECK-%os-DOWNCAST --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t mV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --check-prefix=CHECK-%os-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t fV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t cV 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --check-prefix=CHECK-%os-DOWNCAST --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t oU 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET --check-prefix=CHECK-%os-OFFSET --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t m0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL-MEMBER --check-prefix=CHECK-%os-NULL-MEMBER --strict-whitespace

// RUN: (echo "vptr_check:S"; echo "vptr_check:T"; echo "vptr_check:U") > %t.supp
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t mS
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t fS
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t cS
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t mV
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t fV
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t cV
// RUN: %env_ubsan_opts=suppressions='"%t.supp"' %run %t oU

// RUN: echo "vptr_check:S" > %t.loc-supp
// RUN: %env_ubsan_opts=suppressions='"%t.loc-supp"' not %run %t x- 2>&1 | FileCheck %s --check-prefix=CHECK-LOC-SUPPRESS

// REQUIRES: stable-runtime, cxxabi
#include <new>
#include <assert.h>
#include <stdio.h>

struct S {
  S() : a(0) {}
  ~S() {}
  int a;
  int f() { return 0; }
  virtual int v() { return 0; }
};

struct T : S {
  T() : b(0) {}
  int b;
  int g() { return 0; }
  virtual int v() { return 1; }
};

struct U : S, T { virtual int v() { return 2; } };

struct V : S {};

// Make p global so that lsan does not complain.
T *p = 0;

volatile void *sink1, *sink2;

int access_p(T *p, char type);

int main(int argc, char **argv) {
  assert(argc > 1);
  fprintf(stderr, "Test case: %s\n", argv[1]);
  T t;
  (void)t.a;
  (void)t.b;
  (void)t.f();
  (void)t.g();
  (void)t.v();
  (void)t.S::v();

  U u;
  (void)u.T::a;
  (void)u.b;
  (void)u.T::f();
  (void)u.g();
  (void)u.v();
  (void)u.T::v();
  (void)((T&)u).S::v();

  char Buffer[sizeof(U)] = {};
  char TStorage[sizeof(T)];
  // Allocate two dummy objects so that the real object
  // is not on the boundary of mapped memory. Otherwise ubsan
  // will not be able to describe the vptr in detail.
  sink1 = new T;
  sink2 = new U;
  switch (argv[1][1]) {
  case '0':
    p = reinterpret_cast<T*>(Buffer);
    break;
  case 'S':
    // Make sure p points to the memory chunk of sufficient size to prevent ASan
    // reports about out-of-bounds access.
    p = reinterpret_cast<T*>(new(TStorage) S);
    break;
  case 'T':
    p = new T;
    break;
  case 'U':
    p = new U;
    break;
  case 'V':
    p = reinterpret_cast<T*>(new U);
    break;
  }

  access_p(p, argv[1][0]);
  return 0;
}

int access_p(T *p, char type) {
  switch (type) {
  case 'r':
    // Binding a reference to storage of appropriate size and alignment is OK.
    {T &r = *p;}
    return 0;

  case 'x':
    for (int i = 0; i < 2; i++) {
      // Check that the first iteration ("S") succeeds, while the second ("V") fails.
      p = reinterpret_cast<T*>((i == 0) ? new S : new V);
      // CHECK-LOC-SUPPRESS: vptr.cpp:[[@LINE+5]]:10: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
      // CHECK-LOC-SUPPRESS-NEXT: [[PTR]]: note: object is of type 'V'
      // CHECK-LOC-SUPPRESS-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
      // CHECK-LOC-SUPPRESS-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
      // CHECK-LOC-SUPPRESS-NEXT: {{^              vptr for 'V'}}
      p->g();
    }
    return 0;

  case 'm':
    // CHECK-MEMBER: vptr.cpp:[[@LINE+6]]:15: runtime error: member access within address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-MEMBER-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-MEMBER-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-MEMBER-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    // CHECK-Linux-MEMBER: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    return p->b;

    // CHECK-NULL-MEMBER: vptr.cpp:[[@LINE-2]]:15: runtime error: member access within address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-NULL-MEMBER-NEXT: [[PTR]]: note: object has invalid vptr
    // CHECK-NULL-MEMBER-NEXT: {{^  ?.. .. .. ..  ?00 00 00 00  ?00 00 00 00  ?}}
    // CHECK-NULL-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-NULL-MEMBER-NEXT: {{^              invalid vptr}}
    // CHECK-Linux-NULL-MEMBER: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE-7]]

  case 'f':
    // CHECK-MEMFUN: vptr.cpp:[[@LINE+6]]:15: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-MEMFUN-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-MEMFUN-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-MEMFUN-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-MEMFUN-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    // TODO: Add check for stacktrace here.
    return p->g();

  case 'o':
    // CHECK-OFFSET: vptr.cpp:[[@LINE+6]]:37: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'U'
    // CHECK-OFFSET-NEXT: 0x{{[0-9a-f]*}}: note: object is base class subobject at offset {{8|16}} within object of type [[DYN_TYPE:'U']]
    // CHECK-OFFSET-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  .. .. .. .. .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-OFFSET-NEXT: {{^              \^                        (                         ~~~~~~~~~~~~)?~~~~~~~~~~~ *$}}
    // CHECK-OFFSET-NEXT: {{^                                       (                         )?vptr for}} 'T' base class of [[DYN_TYPE]]
    // CHECK-Linux-OFFSET: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    return reinterpret_cast<U*>(p)->v() - 2;

  case 'c':
    // CHECK-DOWNCAST: vptr.cpp:[[@LINE+6]]:11: runtime error: downcast of address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-DOWNCAST-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-DOWNCAST-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-DOWNCAST-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DOWNCAST-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    // CHECK-Linux-DOWNCAST: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    (void)static_cast<T*>(reinterpret_cast<S*>(p));
    return 0;
  }
  return 0;
}
