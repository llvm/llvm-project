// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -std=c++11 -DPEDANTIC -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -std=c++11 -DPEDANTIC -verify %s -DHEAP_ALLOCATION

#ifdef HEAP_ALLOCATION
#define INIT(CLS, ARGS) new CLS ARGS
#else
#define INIT(CLS, ARGS) (void) CLS ARGS
#endif

class UninitPointerTest {
  int *ptr; // expected-note{{uninitialized pointer 'this->ptr'}}
  int dontGetFilteredByNonPedanticMode = 0;

public:
  UninitPointerTest() {} // expected-warning{{1 uninitialized field}}
};

void fUninitPointerTest() {
  INIT(UninitPointerTest, ());
}

class UninitPointeeTest {
  int *ptr; // no-note
  int dontGetFilteredByNonPedanticMode = 0;

public:
  UninitPointeeTest(int *ptr) : ptr(ptr) {} // no-warning
};

void fUninitPointeeTest() {
  int a;
  INIT(UninitPointeeTest, (&a));
}
