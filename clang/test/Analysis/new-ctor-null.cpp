// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;

void *operator new(size_t size) throw() {
  return nullptr;
}
void *operator new[](size_t size) throw() {
  return nullptr;
}

struct S {
  int x;
  S() : x(1) {}
  ~S() {}
};

void testArrays() {
  S *s = new S[10]; // no-crash
  s[0].x = 2; // expected-warning{{Dereference of null pointer}}
}
