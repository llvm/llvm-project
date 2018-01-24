// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;


// These are ill-formed. One cannot return nullptr from a throwing version of an
// operator new.
void *operator new(size_t size) {
  return nullptr;
}
void *operator new[](size_t size) {
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
