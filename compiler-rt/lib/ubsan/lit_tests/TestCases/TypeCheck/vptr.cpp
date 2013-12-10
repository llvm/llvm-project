// RUN: %clangxx -fsanitize=vptr %s -O3 -o %t
// RUN: %t rT && %t mT && %t fT && %t cT
// RUN: %t rU && %t mU && %t fU && %t cU
// RUN: %t rS && %t rV && %t oV
// RUN: %t mS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --strict-whitespace
// RUN: %t fS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %t cS 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --strict-whitespace
// RUN: %t mV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER --strict-whitespace
// RUN: %t fV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN --strict-whitespace
// RUN: %t cV 2>&1 | FileCheck %s --check-prefix=CHECK-DOWNCAST --strict-whitespace
// RUN: %t oU 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET --strict-whitespace
// RUN: %t m0 2>&1 | FileCheck %s --check-prefix=CHECK-NULL-MEMBER --strict-whitespace

// FIXME: This test produces linker errors on Darwin.
// XFAIL: darwin

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

int main(int, char **argv) {
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

  T *p = 0;
  char Buffer[sizeof(U)] = {};
  switch (argv[1][1]) {
  case '0':
    p = reinterpret_cast<T*>(Buffer);
    break;
  case 'S':
    p = reinterpret_cast<T*>(new S);
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

  switch (argv[1][0]) {
  case 'r':
    // Binding a reference to storage of appropriate size and alignment is OK.
    {T &r = *p;}
    break;

  case 'm':
    // CHECK-MEMBER: vptr.cpp:[[@LINE+5]]:15: runtime error: member access within address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-MEMBER-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-MEMBER-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-MEMBER-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    return p->b;

    // CHECK-NULL-MEMBER: vptr.cpp:[[@LINE-2]]:15: runtime error: member access within address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-NULL-MEMBER-NEXT: [[PTR]]: note: object has invalid vptr
    // CHECK-NULL-MEMBER-NEXT: {{^ .. .. .. ..  00 00 00 00 00 00 00 00  }}
    // CHECK-NULL-MEMBER-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-NULL-MEMBER-NEXT: {{^              invalid vptr}}

  case 'f':
    // CHECK-MEMFUN: vptr.cpp:[[@LINE+5]]:12: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-MEMFUN-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-MEMFUN-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-MEMFUN-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-MEMFUN-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    return p->g();

  case 'o':
    // CHECK-OFFSET: vptr.cpp:[[@LINE+5]]:12: runtime error: member call on address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'U'
    // CHECK-OFFSET-NEXT: 0x{{[0-9a-f]*}}: note: object is base class subobject at offset {{8|16}} within object of type [[DYN_TYPE:'U']]
    // CHECK-OFFSET-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  .. .. .. .. .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-OFFSET-NEXT: {{^              \^                        (                         ~~~~~~~~~~~~)?~~~~~~~~~~~ *$}}
    // CHECK-OFFSET-NEXT: {{^                                       (                         )?vptr for}} 'T' base class of [[DYN_TYPE]]
    return reinterpret_cast<U*>(p)->v() - 2;

  case 'c':
    // CHECK-DOWNCAST: vptr.cpp:[[@LINE+5]]:5: runtime error: downcast of address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
    // CHECK-DOWNCAST-NEXT: [[PTR]]: note: object is of type [[DYN_TYPE:'S'|'U']]
    // CHECK-DOWNCAST-NEXT: {{^ .. .. .. ..  .. .. .. .. .. .. .. ..  }}
    // CHECK-DOWNCAST-NEXT: {{^              \^~~~~~~~~~~(~~~~~~~~~~~~)? *$}}
    // CHECK-DOWNCAST-NEXT: {{^              vptr for}} [[DYN_TYPE]]
    static_cast<T*>(reinterpret_cast<S*>(p));
    return 0;
  }
}
