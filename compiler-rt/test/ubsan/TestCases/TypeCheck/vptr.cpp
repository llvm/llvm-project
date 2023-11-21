// RUN: %clangxx -frtti -fsanitize=null,vptr -fno-sanitize-memory-param-retval -g %s -O3 -o %t -mllvm -enable-tail-merge=false
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t rT
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t mT
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t fT
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t cT
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t rU
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t mU
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t fU
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t cU
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t rS
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t rV
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t oV
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t zN
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t mS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --allow-unused-prefixes --check-prefix=CHECK-%os-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t fS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t cS 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --allow-unused-prefixes --check-prefix=CHECK-%os-DOWNCAST --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t mV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --allow-unused-prefixes --check-prefix=CHECK-%os-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t fV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t cV 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --allow-unused-prefixes --check-prefix=CHECK-%os-DOWNCAST --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t oU 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET --allow-unused-prefixes --check-prefix=CHECK-%os-OFFSET --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t m0 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-MEMBER --allow-unused-prefixes --check-prefix=CHECK-%os-NULL-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1:print_stacktrace=1 not %run %t m0 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-MEMBER --allow-unused-prefixes --check-prefix=CHECK-%os-NULL-MEMBER --strict-whitespace
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t nN 2>&1 | FileCheck %s --check-prefix=CHECK-NULL-MEMFUN --strict-whitespace
// RUN: %env_ubsan_opts=print_stacktrace=1 %run %t dT 2>&1 | FileCheck %s --check-prefix=CHECK-DYNAMIC --allow-unused-prefixes --check-prefix=CHECK-%os-DYNAMIC --strict-whitespace

// RUN: (echo "vptr_check:S"; echo "vptr_check:T"; echo "vptr_check:U") > %t.supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t mS
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t fS
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t cS
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t mV
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t fV
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t cV
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t oU
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.supp"' %run %t dT

// RUN: echo "vptr_check:S" > %t.loc-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.loc-supp"' not %run %t x- 2>&1 | FileCheck %s --check-prefix=CHECK-LOC-SUPPRESS

// REQUIRES: stable-runtime, cxxabi
// UNSUPPORTED: target={{.*windows-msvc.*}}
// Suppressions file not pushed to the device.
// UNSUPPORTED: android
// Compilation error
// UNSUPPORTED: target={{.*openbsd.*}}
// Compilation error
// UNSUPPORTED: target={{.*freebsd.*}}
// FIXME: For MinGW targets, the vptr tests do generally work, but Itanium
// demangling isn't done for the type names. The "(echo ..." line fails to
// be handled by the shell.
// XFAIL: target={{.*windows-gnu.*}}
#include <new>
#include <typeinfo>
#include <assert.h>
#include <stdio.h>

struct S {
  S() : a(0) {}
  ~S();
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

namespace {
  struct W {};
}

T *p = 0;

bool dtorCheck = false;

volatile void *sink1, *sink2;

int access_p(T *p, char type);

S::~S() {
  if (dtorCheck)
    access_p(p, '~');
}

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
  case 'N':
    p = 0;
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
    // CHECK-MEMBER-NEXT: {{^  ?.. .. .. ..  ?.. .. .. ..  ?.. .. .. ..  ?}}
    // CHECK-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-MEMBER-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    // CHECK-Linux-MEMBER: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    return p->b;

    // CHECK-INVALID-MEMBER: vptr.cpp:[[@LINE-2]]:15: runtime error: member access within address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-INVALID-MEMBER-NEXT: [[PTR]]: note: object has invalid vptr
    // CHECK-INVALID-MEMBER-NEXT: {{^  ?.. .. .. ..  ?00 00 00 00  ?00 00 00 00  ?}}
    // CHECK-INVALID-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-INVALID-MEMBER-NEXT: {{^              invalid vptr}}
    // CHECK-Linux-NULL-MEMBER: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE-7]]

  case 'f':
    // CHECK-MEMFUN: vptr.cpp:[[@LINE+6]]:15: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-MEMFUN-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-MEMFUN-NEXT: {{^  ?.. .. .. ..  ?.. .. .. ..  ?.. .. .. ..  ?}}
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
    // CHECK-DOWNCAST-NEXT: {{^  ?.. .. .. ..  ?.. .. .. ..  ?.. .. .. ..  ?}}
    // CHECK-DOWNCAST-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DOWNCAST-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    // CHECK-Linux-DOWNCAST: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    (void)static_cast<T*>(reinterpret_cast<S*>(p));
    return 0;

  case 'n':
    // CHECK-NULL-MEMFUN: vptr.cpp:[[@LINE+1]]:15: runtime error: member call on null pointer of type 'T'
    return p->g();

  case 'd':
    dtorCheck = true;
    delete p;
    dtorCheck = false;
    return 0;
  case '~':
    // CHECK-DYNAMIC: vptr.cpp:[[@LINE+6]]:11: runtime error: dynamic operation on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-DYNAMIC-NEXT: [[PTR]]: note: object is of type 'S'
    // CHECK-DYNAMIC-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-DYNAMIC-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DYNAMIC-NEXT: {{^              vptr for}} 'S'
    // CHECK-Linux-DYNAMIC: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    (void)dynamic_cast<V*>(p);
    // CHECK-DYNAMIC: vptr.cpp:[[@LINE+6]]:11: runtime error: dynamic operation on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-DYNAMIC-NEXT: [[PTR]]: note: object is of type 'S'
    // CHECK-DYNAMIC-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-DYNAMIC-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DYNAMIC-NEXT: {{^              vptr for}} 'S'
    // CHECK-Linux-DYNAMIC: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    (void)dynamic_cast<W*>(p);
    try {
      // CHECK-DYNAMIC: vptr.cpp:[[@LINE+6]]:13: runtime error: dynamic operation on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
      // CHECK-DYNAMIC-NEXT: [[PTR]]: note: object is of type 'S'
      // CHECK-DYNAMIC-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
      // CHECK-DYNAMIC-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
      // CHECK-DYNAMIC-NEXT: {{^              vptr for}} 'S'
      // CHECK-Linux-DYNAMIC: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
      (void)dynamic_cast<V&>(*p);
    } catch (std::bad_cast &) {}
    // CHECK-DYNAMIC: vptr.cpp:[[@LINE+6]]:18: runtime error: dynamic operation on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-DYNAMIC-NEXT: [[PTR]]: note: object is of type 'S'
    // CHECK-DYNAMIC-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-DYNAMIC-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DYNAMIC-NEXT: {{^              vptr for}} 'S'
    // CHECK-Linux-DYNAMIC: #0 {{.*}}access_p{{.*}}vptr.cpp:[[@LINE+1]]
    (void)typeid(*p);
    return 0;

  case 'z':
    (void)dynamic_cast<V*>(p);
    try {
      (void)typeid(*p);
    } catch (std::bad_typeid &) {}
    return 0;
  }
  return 0;
}
