// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core \
// RUN:  -analyzer-config suppress-null-return-paths=false \
// RUN:  -verify %s
// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=core \
// RUN:  -DSUPPRESSED \
// RUN:  -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;


// These are ill-formed. One cannot return nullptr from a throwing version of an
// operator new.
void *operator new(size_t size) {
  return nullptr;
  // expected-warning@-1 {{'operator new' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
}
void *operator new[](size_t size) {
  return nullptr;
  // expected-warning@-1 {{'operator new[]' should not return a null pointer unless it is declared 'throw()' or 'noexcept'}}
}

struct S {
  int x;
  S() : x(1) {}
  ~S() {}
  int getX() const { return x; }
};

void testArrays() {
  S *s = new S[10]; // no-crash
  s[0].x = 2;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testCtor() {
  S *s = new S();
  s->x = 13;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Access to field 'x' results in a dereference of a null pointer (loaded from variable 's')}}
#endif
}

void testMethod() {
  S *s = new S();
  const int X = s->getX();
#ifndef SUPPRESSED
  // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
}

