// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -std=c++11 -verify %s

//===----------------------------------------------------------------------===//
// Tests for union fields inside structs/classes.
//===----------------------------------------------------------------------===//

// A struct with an uninitialized union field -- should warn.
struct WithUninitUnion {
  union {
    int i;
    float f;
  } u; // expected-note{{uninitialized field 'this->u'}}
  int x;

  WithUninitUnion(int val) : x(val) { // expected-warning{{1 uninitialized field at the end of the constructor call}}
    // u is never initialized
  }
};

void fWithUninitUnion() {
  WithUninitUnion w(42);
}

// A struct where the union field IS initialized -- should not warn.
struct WithInitUnion {
  union {
    int i;
    float f;
  } u;
  int x;

  WithInitUnion(int val) : x(val) {
    u.i = val; // union is initialized via one of its members
  }
};

void fWithInitUnion() {
  WithInitUnion w(42); // no-warning
}

// A struct with only a union field, left uninitialized (pedantic mode).
struct OnlyUninitUnion {
  union {
    int i;
    char c;
  } u; // expected-note{{uninitialized field 'this->u'}}

  OnlyUninitUnion() { // expected-warning{{1 uninitialized field at the end of the constructor call}}
    // u is never initialized
  }
};

void fOnlyUninitUnion() {
  OnlyUninitUnion o;
}
