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
