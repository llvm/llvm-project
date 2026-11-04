// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject -std=c++11 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject -std=c++11 -fblocks -verify %s -DHEAP_ALLOCATION

#ifdef HEAP_ALLOCATION
#define INIT(CLS, ARGS) new CLS ARGS
#else
#define INIT(CLS, ARGS) (void) CLS ARGS
#endif

typedef void (^myBlock) ();

struct StructWithBlock {
  int a;
  myBlock z; // expected-note{{uninitialized field 'this->z'}}

  StructWithBlock() : a(0), z(^{}) {}

  // Miss initialization of field `z`.
  StructWithBlock(int pA) : a(pA) {} // expected-warning{{1 uninitialized field at the end of the constructor call}}

};

void warnOnUninitializedBlock() {
  INIT(StructWithBlock, (10));
}

void noWarningWhenInitialized() {
  INIT(StructWithBlock, ());
}

struct StructWithId {
  int a;
  id z; // expected-note{{uninitialized pointer 'this->z'}}
  StructWithId() : a(0) {} // expected-warning{{1 uninitialized field at the end of the constructor call}}
};

void warnOnUninitializedId() {
  INIT(StructWithId, ());
}
