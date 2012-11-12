// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-store region -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);

int someGlobal;
void testImplicitlyDeclaredGlobalNew() {
  if (someGlobal != 0)
    return;

  // This used to crash because the global operator new is being implicitly
  // declared and it does not have a valid source location. (PR13090)
  void *x = ::operator new(0);
  ::operator delete(x);

  // Check that the new/delete did not invalidate someGlobal;
  clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
}


// This is the standard placement new.
inline void* operator new(size_t, void* __p) throw()
{
  return __p;
}

void *testPlacementNew() {
  int *x = (int *)malloc(sizeof(int));
  *x = 1;
  clang_analyzer_eval(*x == 1); // expected-warning{{TRUE}};

  void *y = new (x) int;
  clang_analyzer_eval(x == y); // expected-warning{{TRUE}};
  clang_analyzer_eval(*x == 1); // expected-warning{{UNKNOWN}};

  return y;
}

void *operator new(size_t, size_t, int *);
void *testCustomNew() {
  int x[1] = {1};
  clang_analyzer_eval(*x == 1); // expected-warning{{TRUE}};

  void *y = new (0, x) int;
  clang_analyzer_eval(*x == 1); // expected-warning{{UNKNOWN}};

  return y; // no-warning
}

void *operator new(size_t, void *, void *);
void *testCustomNewMalloc() {
  int *x = (int *)malloc(sizeof(int));

  // Should be no-warning (the custom allocator could have freed x).
  void *y = new (0, x) int; // no-warning

  return y;
}

void testScalarInitialization() {
  int *n = new int(3);
  clang_analyzer_eval(*n == 3); // expected-warning{{TRUE}}

  new (n) int();
  clang_analyzer_eval(*n == 0); // expected-warning{{TRUE}}

  new (n) int{3};
  clang_analyzer_eval(*n == 3); // expected-warning{{TRUE}}

  new (n) int{};
  clang_analyzer_eval(*n == 0); // expected-warning{{TRUE}}
}


struct PtrWrapper {
  int *x;

  PtrWrapper(int *input) : x(input) {}
};

PtrWrapper *testNewInvalidation() {
  // Ensure that we don't consider this a leak.
  return new PtrWrapper(static_cast<int *>(malloc(4)));
}


//--------------------------------
// Incorrectly-modelled behavior
//--------------------------------

int testNoInitialization() {
  int *n = new int;

  // Should warn that *n is uninitialized.
  if (*n) { // no-warning
    return 0;
  }
  return 1;
}

int testNoInitializationPlacement() {
  int n;
  new (&n) int;

  // Should warn that n is uninitialized.
  if (n) { // no-warning
    return 0;
  }
  return 1;
}
