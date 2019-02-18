// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UninitializedObject \
// RUN:   -std=c++11 -DPEDANTIC -verify %s

class UninitPointerTest {
  int *ptr; // expected-note{{uninitialized pointer 'this->ptr'}}
  int dontGetFilteredByNonPedanticMode = 0;

public:
  UninitPointerTest() {} // expected-warning{{1 uninitialized field}}
};

void fUninitPointerTest() {
  UninitPointerTest();
}

class UninitPointeeTest {
  int *ptr; // no-note
  int dontGetFilteredByNonPedanticMode = 0;

public:
  UninitPointeeTest(int *ptr) : ptr(ptr) {} // no-warning
};

void fUninitPointeeTest() {
  int a;
  UninitPointeeTest t(&a);
}
