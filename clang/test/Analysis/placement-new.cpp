// RUN: %clang_analyze_cc1 -std=c++11 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=cplusplus.PlacementNew \
// RUN:   -analyzer-output=text -verify \
// RUN:   -triple x86_64-unknown-linux-gnu

#include "Inputs/system-header-simulator-cxx.h"

void f() {
  short s;                    // expected-note {{'s' declared without an initial value}}
  long *lp = ::new (&s) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 3 {{}}
  (void)lp;
}

namespace testArrayNew {
void f() {
  short s;                        // expected-note {{'s' declared without an initial value}}
  char *buf = ::new (&s) char[8]; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 3 {{}}
  (void)buf;
}
} // namespace testArrayNew

namespace testBufferInOtherFun {
void f(void *place) {
  long *lp = ::new (place) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
void g() {
  short buf; // expected-note {{'buf' declared without an initial value}}
  f(&buf);   // expected-note 2 {{}}
}
} // namespace testBufferInOtherFun

namespace testArrayBuffer {
void f(void *place) {
  long *lp = ::new (place) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
void g() {
  char buf[2]; // expected-note {{'buf' initialized here}}
  f(&buf);     // expected-note 2 {{}}
}
} // namespace testArrayBuffer

namespace testGlobalPtrAsPlace {
void *gptr = nullptr;
short gs;
void f() {
  gptr = &gs; // expected-note {{Value assigned to 'gptr'}}
}
void g() {
  f();                          // expected-note 2 {{}}
  long *lp = ::new (gptr) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testGlobalPtrAsPlace

namespace testRvalue {
short gs;
void *f() {
  return &gs;
}
void g() {
  long *lp = ::new (f()) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testRvalue

namespace testNoWarning {
void *f();
void g() {
  long *lp = ::new (f()) long;
  (void)lp;
}
} // namespace testNoWarning

namespace testPtrToArrayAsPlace {
void f() {
  //char *st = new char [8];
  char buf[3];                // expected-note {{'buf' initialized here}}
  void *st = buf;             // expected-note {{'st' initialized here}}
  long *lp = ::new (st) long; // expected-warning{{Storage provided to placement new is only 3 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testPtrToArrayAsPlace

namespace testPtrToArrayWithOffsetAsPlace {
void f() {
  int buf[3];                      // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf + 2) long; // expected-warning{{Storage provided to placement new is only 4 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testPtrToArrayWithOffsetAsPlace

namespace testZeroSize {
void f() {
  int buf[3];                      // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf + 3) long; // expected-warning{{Storage provided to placement new is only 0 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testZeroSize

namespace testNegativeSize {
void f() {
  int buf[3];                      // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf + 4) long; // expected-warning{{Storage provided to placement new is only -4 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testNegativeSize

namespace testHeapAllocatedBuffer {
void g2() {
  char *buf = new char[2];     // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testHeapAllocatedBuffer

namespace testMultiDimensionalArray {
void f() {
  char buf[2][3];              // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf) long; // expected-warning{{Storage provided to placement new is only 6 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testMultiDimensionalArray

namespace testMultiDimensionalArray2 {
void f() {
  char buf[2][3];                  // expected-note {{'buf' initialized here}}
  long *lp = ::new (buf + 1) long; // expected-warning{{Storage provided to placement new is only 3 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testMultiDimensionalArray2

namespace testMultiDimensionalArray3 {
void f() {
  char buf[2][3];                     // expected-note {{'buf' initialized here}}
  long *lp = ::new (&buf[1][1]) long; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)lp;
}
} // namespace testMultiDimensionalArray3

namespace testHierarchy {
struct Base {
  char a[2];
};
struct Derived : Base {
  char x[2];
  int y;
};
void f() {
  Base b;                           // expected-note {{'b' initialized here}}
  Derived *dp = ::new (&b) Derived; // expected-warning{{Storage provided to placement new is only 2 bytes, whereas the allocated type requires 8 bytes}} expected-note 1 {{}}
  (void)dp;
}
} // namespace testHierarchy

namespace testArrayTypesAllocation {
void f1() {
  struct S {
    short a;
  };

  // bad (not enough space).
  const unsigned N = 32;
  alignas(S) unsigned char buffer1[sizeof(S) * N]; // expected-note {{'buffer1' initialized here}}
  ::new (buffer1) S[N];                            // expected-warning{{Storage provided to placement new is only 64 bytes, whereas the allocated array type requires more space for internal needs}} expected-note 1 {{}}
}

void f2() {
  struct S {
    short a;
  };

  // maybe ok but we need to warn.
  const unsigned N = 32;
  alignas(S) unsigned char buffer2[sizeof(S) * N + sizeof(int)]; // expected-note {{'buffer2' initialized here}}
  ::new (buffer2) S[N];                                          // expected-warning{{68 bytes is possibly not enough for array allocation which requires 64 bytes. Current overhead requires the size of 4 bytes}} expected-note 1 {{}}
}
} // namespace testArrayTypesAllocation

namespace testStructAlign {
void f1() {
  struct X {
    char a[9];
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad (struct X is aligned to char).
  ::new (&Xi.a) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f2() {
  struct X {
    char a;
    char b;
    long c;
  } Xi;

  // ok (struct X is aligned to long).
  ::new (&Xi.a) long;
}

void f3() {
  struct X {
    char a;
    char b;
    long c;
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad (struct X is aligned to long but field 'b' is aligned to 1 because of its offset)
  ::new (&Xi.b) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f4() {
  struct X {
    char a;
    struct alignas(alignof(short)) Y {
      char b;
      char c;
    } y;
    long d;
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad. 'b' is aligned to short
  ::new (&Xi.y.b) long; // expected-warning{{Storage type is aligned to 2 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f5() {
  short b[10]; // expected-note {{'b' initialized here}}

  ::new (&b) long; // expected-warning{{Storage type is aligned to 2 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f6() {
  short b[10]; // expected-note {{'b' initialized here}}

  // bad (same as previous but checks ElementRegion case)
  ::new (&b[0]) long; // expected-warning{{Storage type is aligned to 2 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f7() {
  alignas(alignof(long)) short b[10];

  // ok. aligned to long(ok). offset 4*2(ok)
  ::new (&b[4]) long;
}

void f8() {
  alignas(alignof(long)) short b[10]; // expected-note {{'b' initialized here}}

  // ok. aligned to long(ok). offset 3*2(ok)
  ::new (&b[3]) long; // expected-warning{{Storage type is aligned to 6 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f9() {
  struct X {
    char a;
    alignas(alignof(long)) char b[20];
  } Xi; // expected-note {{'Xi' initialized here}}

  // ok. aligned to long(ok). offset 8*1(ok)
  ::new (&Xi.b[8]) long;

  // bad. aligned to long(ok). offset 1*1(ok)
  ::new (&Xi.b[1]) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f10() {
  struct X {
    char a[8];
    alignas(2) char b;
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad (struct X is aligned to 2).
  ::new (&Xi.a) long; // expected-warning{{Storage type is aligned to 2 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void f11() {
  struct X {
    char a;
    char b;
    struct Y {
      long c;
    } d;
  } Xi;

  // ok (struct X is aligned to long).
  ::new (&Xi.a) long;
}

void f12() {
  struct alignas(alignof(long)) X {
    char a;
    char b;
  } Xi;

  // ok (struct X is aligned to long).
  ::new (&Xi.a) long;
}

void test13() {
  struct Y {
    char a[10];
  };

  struct X {
    Y b[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad. X,A are aligned to 'char'
  ::new (&Xi.b[0].a) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void test14() {
  struct Y {
    char a[10];
  };

  struct alignas(alignof(long)) X {
    Y b[10];
  } Xi;

  // ok. X is aligned to 'long' and field 'a' goes with zero offset
  ::new (&Xi.b[0].a) long;
}

void test15() {
  struct alignas(alignof(long)) Y {
    char a[10];
  };

  struct X {
    Y b[10];
  } Xi;

  // ok. X is aligned to 'long' because it contains struct 'Y' which is aligned to 'long'
  ::new (&Xi.b[0].a) long;
}

void test16() {
  struct alignas(alignof(long)) Y {
    char p;
    char a[10];
  };

  struct X {
    Y b[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad. aligned to long(ok). offset 1(bad)
  ::new (&Xi.b[0].a) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void test17() {
  struct alignas(alignof(long)) Y {
    char p;
    char a[10];
  };

  struct X {
    Y b[10];
  } Xi;

  // ok. aligned to long(ok). offset 1+7*1(ok)
  ::new (&Xi.b[0].a[7]) long;
}

void test18() {
  struct Y {
    char p;
    alignas(alignof(long)) char a[10];
  };

  struct X {
    Y b[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // ok. aligned to long(ok). offset 8*1(ok)
  ::new (&Xi.b[0].a[8]) long;

  // bad. aligned to long(ok). offset 1(bad)
  ::new (&Xi.b[0].a[1]) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void test19() {
  struct Z {
    char p;
    char c[10];
  };

  struct Y {
    char p;
    Z b[10];
  };

  struct X {
    Y a[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // bad. all structures X,Y,Z are aligned to char
  ::new (&Xi.a[1].b[1].c) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void test20() {
  struct Z {
    char p;
    alignas(alignof(long)) char c[10];
  };

  struct Y {
    char p;
    Z b[10];
  };

  struct X {
    Y a[10];
  } Xi;

  // ok. field 'c' is aligned to 'long'
  ::new (&Xi.a[1].b[1].c) long;
}

void test21() {
  struct Z {
    char p;
    char c[10];
  };

  struct Y {
    char p;
    Z b[10];
  };

  struct alignas(alignof(long)) X {
    Y a[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // ok. aligned to long(ok). offset 1+7*1(ok)
  ::new (&Xi.a[0].b[0].c[7]) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

void test22() {
  struct alignas(alignof(long)) Y {
    char p;
    char a[10][10];
  };

  struct X {
    Y b[10];
  } Xi; // expected-note {{'Xi' initialized here}}

  // ok. aligned to long(ok). offset ok. 1(field 'a' offset) + 0*10(index '0' * first dimension size '10') + 7*1(index '7')
  ::new (&Xi.b[0].a[0][7]) long;

  // ok. aligned to long(ok). offset ok. 1(field 'a' offset) + 1*10(index '1' * first dimension size '10') + 5*1(index '5')
  ::new (&Xi.b[0].a[1][5]) long;

  // bad. aligned to long(ok). offset ok. 1(field 'a' offset) + 1*10(index '1' * first dimension size '10') + 6*1(index '5')
  ::new (&Xi.b[0].a[1][6]) long; // expected-warning{{Storage type is aligned to 1 bytes but allocated type is aligned to 8 bytes}} expected-note 1 {{}}
}

} // namespace testStructAlign
