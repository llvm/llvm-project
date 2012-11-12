// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store region -std=c++11 -fexceptions -fcxx-exceptions -verify -DEXCEPTIONS %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store region -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);

// This is the standard placement new.
inline void* operator new(size_t, void* __p) throw()
{
  return __p;
}

struct NoThrow {
  void *operator new(size_t) throw();
};

struct NoExcept {
  void *operator new(size_t) noexcept;
};

struct DefaultThrow {
  void *operator new(size_t);
};

struct ExplicitThrow {
  void *operator new(size_t) throw(int);
};

void testNew() {
  clang_analyzer_eval(new NoThrow); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new NoExcept); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(new DefaultThrow);
  clang_analyzer_eval(new ExplicitThrow);
#ifdef EXCEPTIONS
  // expected-warning@-3 {{TRUE}}
  // expected-warning@-3 {{TRUE}}
#else
  // expected-warning@-6 {{UNKNOWN}}
  // expected-warning@-6 {{UNKNOWN}}
#endif
}

void testNewArray() {
  clang_analyzer_eval(new NoThrow[2]);
  clang_analyzer_eval(new NoExcept[2]);
  clang_analyzer_eval(new DefaultThrow[2]);
  clang_analyzer_eval(new ExplicitThrow[2]);
#ifdef EXCEPTIONS
  // expected-warning@-5 {{TRUE}}
  // expected-warning@-5 {{TRUE}}
  // expected-warning@-5 {{TRUE}}
  // expected-warning@-5 {{TRUE}}
#else
  // expected-warning@-10 {{UNKNOWN}}
  // expected-warning@-10 {{UNKNOWN}}
  // expected-warning@-10 {{UNKNOWN}}
  // expected-warning@-10 {{UNKNOWN}}
#endif
}

extern void *operator new[](size_t, int) noexcept;

void testNewArrayNoThrow() {
  clang_analyzer_eval(new (1) NoThrow[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) NoExcept[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) DefaultThrow[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) ExplicitThrow[2]); // expected-warning{{UNKNOWN}}
}
