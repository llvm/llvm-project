// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

void usePointer(int * const *);
void useReference(int * const &);

void testPointer() {
  int x;
  int *p;

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp1 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&cp1);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp2 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(cp2);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct Wrapper {
  int *ptr;
};

void useStruct(Wrapper &w);
void useConstStruct(const Wrapper &w);

void testPointerStruct() {
  int x;
  Wrapper w;

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct RefWrapper {
  int &ref;
};

void useStruct(RefWrapper &w);
void useConstStruct(const RefWrapper &w);

void testReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}

// FIXME: This test is split into two functions because region invalidation
// does not preserve reference bindings.
void testConstReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


void usePointerPure(int * const *) __attribute__((pure));
void usePointerConst(int * const *) __attribute__((const));

void testPureConst() {
  extern int global;
  int x;
  int *p;

  p = &x;
  x = 42;
  global = -5;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointerPure(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointerConst(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(global == -5); // expected-warning{{UNKNOWN}}
}


struct PlainStruct {
  int x, y;
  mutable int z;
};

PlainStruct glob;

void useAnything(void *);
void useAnythingConst(const void *);

void testInvalidationThroughBaseRegionPointer() {
  PlainStruct s1;
  s1.x = 1;
  s1.z = 1;
  clang_analyzer_eval(s1.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(s1.z == 1); // expected-warning{{TRUE}}
  // Not only passing a structure pointer through const pointer parameter,
  // but also passing a field pointer through const pointer parameter
  // should preserve the contents of the structure.
  useAnythingConst(&(s1.y));
  clang_analyzer_eval(s1.x == 1); // expected-warning{{TRUE}}
  // FIXME: Should say "UNKNOWN", because it is not uncommon to
  // modify a mutable member variable through const pointer.
  clang_analyzer_eval(s1.z == 1); // expected-warning{{TRUE}}
  useAnything(&(s1.y));
  clang_analyzer_eval(s1.x == 1); // expected-warning{{UNKNOWN}}
}


void useFirstConstSecondNonConst(const void *x, void *y);
void useFirstNonConstSecondConst(void *x, const void *y);

void testMixedConstNonConstCalls() {
  PlainStruct s2;
  s2.x = 1;
  useFirstConstSecondNonConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.x == 1); // expected-warning{{UNKNOWN}}
  s2.x = 1;
  useFirstNonConstSecondConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.x == 1); // expected-warning{{UNKNOWN}}
  s2.y = 1;
  useFirstConstSecondNonConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.y == 1); // expected-warning{{UNKNOWN}}
  s2.y = 1;
  useFirstNonConstSecondConst(&(s2.x), &(s2.y));
  clang_analyzer_eval(s2.y == 1); // expected-warning{{UNKNOWN}}
}
