// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject -std=c++11 -fblocks -verify %s

typedef void (^myBlock) ();

struct StructWithBlock {
  int a;
  myBlock z; // expected-note{{uninitialized field 'this->z'}}

  StructWithBlock() : a(0), z(^{}) {}

  // Miss initialization of field `z`.
  StructWithBlock(int pA) : a(pA) {} // expected-warning{{1 uninitialized field at the end of the constructor call}}

};

void warnOnUninitializedBlock() {
  StructWithBlock a(10);
}

void noWarningWhenInitialized() {
  StructWithBlock a;
}

struct StructWithId {
  int a;
  id z; // expected-note{{uninitialized pointer 'this->z'}}
  StructWithId() : a(0) {} // expected-warning{{1 uninitialized field at the end of the constructor call}}
};

void warnOnUninitializedId() {
  StructWithId s;
}
