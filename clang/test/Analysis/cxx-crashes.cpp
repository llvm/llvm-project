// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -verify %s
// REQUIRES: LP64

void clang_analyzer_eval(bool);

int f1(char *dst) {
  char *p = dst + 4;
  char *q = dst + 3;
  return !(q >= p);
}

long f2(char *c) {
  return long(c) & 1;
}

bool f3() {
  return !false;
}

void *f4(int* w) {
  return reinterpret_cast<void*&>(w);
}

namespace {

struct A { };
struct B {
  operator A() { return A(); }
};

A f(char *dst) {
  B b;
  return b;
}

}

namespace {

struct S {
    void *p;
};

void *f(S* w) {
    return &reinterpret_cast<void*&>(*w);
}

}

namespace {

struct C { 
  void *p;
  static void f();
};

void C::f() { }

}


void vla(int n) {
  int nums[n];
  nums[0] = 1;
  clang_analyzer_eval(nums[0] == 1); // expected-warning{{TRUE}}
  
  // This used to fail with MallocChecker on, and /only/ in C++ mode.
  // This struct is POD, though, so it should be fine to put it in a VLA.
  struct { int x; } structs[n];
  structs[0].x = 1;
  clang_analyzer_eval(structs[0].x == 1); // expected-warning{{TRUE}}
}

void useIntArray(int []);
void testIntArrayLiteral() {
  useIntArray((int []){ 1, 2, 3 });
}

