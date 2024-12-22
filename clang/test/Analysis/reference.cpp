// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -Wno-null-dereference -Wno-tautological-undefined-compare -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

typedef typeof(sizeof(int)) size_t;
void malloc (size_t);

void f1() {
  int const &i = 3;
  int b = i;

  int *p = 0;

  if (b != 3)
    *p = 1; // no-warning
}

char* ptr();
char& ref();

// These next two tests just shouldn't crash.
char t1 () {
  ref() = 'c';
  return '0';
}

// just a basic correctness test, the same behavior as t1()
char t2 () {
  *ptr() = 'c';
  return '0';
}

// Each of the tests below is repeated with pointers as well as references.
// This is mostly a basic correctness check, but then again, both should work!
char t3 () {
  char& r = ref();
  r = 'c'; // no-warning
  if (r) return r;
  return *(char*)0; // no-warning
}

char t4 () {
  char* p = ptr();
  *p = 'c'; // no-warning
  if (*p) return *p;
  return *(char*)0; // no-warning
}

char t5 (char& r) {
  r = 'c'; // no-warning
  if (r) return r;
  return *(char*)0; // no-warning
}

char t6 (char* p) {
  *p = 'c'; // no-warning
  if (*p) return *p;
  return *(char*)0; // no-warning
}


// PR13440
// Test that the array-to-pointer decay works for array references as well.
// More generally, when we want an lvalue for a reference field, we still need
// to do one level of load.
namespace PR13440 {
  typedef int T[1];
  struct S {
    T &x;

    int *m() { return x; }
  };

  struct S2 {
    int (&x)[1];

    int *m() { return x; }

    void testArrayToPointerDecayWithNonTypedValueRegion() {
      int *p = x;
      int *q = x;
      clang_analyzer_eval(p[0] == q[0]); // expected-warning{{TRUE}}
    }

  };

  void test() {
    int a[1];
    S s = { a };
    S2 s2 = { a };

    if (s.x != a) return;  // expected-warning {{comparison between two arrays}}
    if (s2.x != a) return; // expected-warning {{comparison between two arrays}}

    a[0] = 42;
    clang_analyzer_eval(s.x[0] == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(s2.x[0] == 42); // expected-warning{{TRUE}}
  }
}

void testNullReference() {
  int *x = 0;
  int &y = *x; // expected-warning{{Dereference of null pointer}}
  y = 5;
}

void testRetroactiveNullReference(int *x) {
  // According to the C++ standard, there is no such thing as a
  // "null reference". So the 'if' statement ought to be dead code.
  // However, Clang (and other compilers) don't actually check that a pointer
  // value is non-null in the implementation of references, so it is possible
  // to produce a supposed "null reference" at runtime. The analyzer should
  // still warn when it can prove such errors.
  int &y = *x;
  if (x != 0)
    return;
  y = 5; // expected-warning{{Dereference of null pointer}}
}

namespace TestReferenceAddress {
struct S { int &x; };
S getS();
S *getSP();

void testReferenceAddress(int &x) {
// FIXME: Move non-zero reference assumption out of RangeConstraintManager.cpp:422
#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval(&x != 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&ref() != 0); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval(&x != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&ref() != 0); // expected-warning{{TRUE}}
#endif

#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval(&getS().x != 0); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval(&getS().x != 0); // expected-warning{{TRUE}}
#endif

#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval(&getSP()->x != 0); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval(&getSP()->x != 0); // expected-warning{{TRUE}}
#endif
}
}

void testFunctionPointerReturn(void *opaque) {
  typedef int &(*RefFn)();

  RefFn getRef = (RefFn)opaque;

  // Don't crash writing to or reading from this reference.
  int &x = getRef();
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
}

int &testReturnNullReference() {
  int *x = 0;
  return *x; // expected-warning{{Returning null reference}}
}

char &refFromPointer() {
  return *ptr();
}

void testReturnReference() {
  clang_analyzer_eval(ptr() == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&refFromPointer() == 0); // expected-warning{{FALSE}}
}

void intRefParam(int &r) {
	;
}

void test(int *ptr) {
	clang_analyzer_eval(ptr == 0); // expected-warning{{UNKNOWN}}

	extern void use(int &ref);
	use(*ptr);

	clang_analyzer_eval(ptr == 0); // expected-warning{{FALSE}}
}

void testIntRefParam() {
	int i = 0;
	intRefParam(i); // no-warning
}

int refParam(int &byteIndex) {
	return byteIndex;
}

void testRefParam(int *p) {
	if (p)
		;
	refParam(*p); // expected-warning {{Forming reference to null pointer}}
}

int ptrRefParam(int *&byteIndex) {
	return *byteIndex;  // expected-warning {{Dereference of null pointer}}
}
void testRefParam2() {
	int *p = 0;
	int *&rp = p;
	ptrRefParam(rp);
}

int *maybeNull() {
	extern bool coin();
	static int x;
	return coin() ? &x : 0;
}

void use(int &x) {
	x = 1; // no-warning
}

void testSuppression() {
	use(*maybeNull());
}

namespace rdar11212286 {
  class B{};

  B test() {
    B *x = 0;
    return *x; // expected-warning {{Forming reference to null pointer}}
  }

  B testif(B *x) {
    if (x)
      ;
    return *x; // expected-warning {{Forming reference to null pointer}}
  }

  void idc(B *x) {
    if (x)
      ;
  }

  B testidc(B *x) {
    idc(x);
    return *x; // no-warning
  }
}

namespace PR15694 {
  class C {
    bool bit : 1;
    template <class T> void bar(const T &obj) {}
    void foo() {
      bar(bit); // don't crash
    }
  };
}
