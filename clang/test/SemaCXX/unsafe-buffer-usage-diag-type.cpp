// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions -verify %s

namespace localVar {
void testRefersPtrLocalVarDecl(int i) {
  int * ptr;    // expected-warning{{'ptr' is an unsafe pointer used for buffer access}}
  ptr + i;      // expected-note{{used in pointer arithmetic here}}
  ptr[i];       // expected-note{{used in buffer access here}}
}

void testRefersArrayLocalVarDecl(int i) {
  int array[i];   // expected-warning{{'array' is an unsafe buffer that does not perform bounds}}
  array[i/2];     // expected-note{{used in buffer access here}}
}
}

namespace globalVar {
int * ptr;      // expected-warning{{'ptr' is an unsafe pointer used for buffer access}}
void testRefersPtrGlobalVarDecl(int i) {
  ptr + i;      // expected-note{{used in pointer arithmetic here}}
  ptr[i];       // expected-note{{used in buffer access here}}
}

int array[10];     // expected-warning{{'array' is an unsafe buffer that does not perform bounds}}
void testRefersArrayGlobalVarDecl(int i) {
  array[i/2];     // expected-note{{used in buffer access here}}
}
}

namespace functionParm {
void testRefersPtrParmVarDecl(int * ptr) {
                // expected-warning@-1{{'ptr' is an unsafe pointer used for buffer access}}
  ptr + 5;      // expected-note{{used in pointer arithmetic here}}
  ptr[5];       // expected-note{{used in buffer access here}}
}

// FIXME: shall we explain the array to pointer decay to make the warning more understandable?
void testRefersArrayParmVarDecl(int array[10]) {
                // expected-warning@-1{{'array' is an unsafe pointer used for buffer access}}
  array[2];     // expected-note{{used in buffer access here}}
}
}

namespace structField {
struct Struct1 {
  int * ptr;      // FIXME: per-declaration warning aggregated at the struct definition?
};

void testRefersPtrStructFieldDecl(int i) {
  Struct1 s1;
  s1.ptr + i;     // expected-warning{{unsafe pointer arithmetic}}
  s1.ptr[i];      // expected-warning{{unsafe buffer access}}
}

struct Struct2 {
  int array[10];  // FIXME: per-declaration warning aggregated at the struct definition?
};

void testRefersArrayStructFieldDecl(int i) {
  Struct2 s2;
  s2.array[i/2];  // expected-warning{{unsafe buffer access}}
}
}

namespace structFieldFromMethod {
struct Struct1 {
  int * ptr;      // FIXME: per-declaration warning aggregated at the struct definition

  void testRefersPtrStructFieldDecl(int i) {
    ptr + i;     // expected-warning{{unsafe pointer arithmetic}}
    ptr[i];      // expected-warning{{unsafe buffer access}}
  }
};

struct Struct2 {
  int array[10];  // FIXME: per-declaration warning aggregated at the struct definition

  void testRefersArrayStructFieldDecl(int i) {
    Struct2 s2;
    s2.array[i/2];  // expected-warning{{unsafe buffer access}}
  }
};
}

namespace staticStructField {
struct Struct1 {
  static int * ptr;      // expected-warning{{'ptr' is an unsafe pointer used for buffer access}}
};

void testRefersPtrStructFieldDecl(int i) {
  Struct1::ptr + i;      // expected-note{{used in pointer arithmetic here}}
  Struct1::ptr[i];       // expected-note{{used in buffer access here}}
}

struct Struct2 {
  static int array[10];     // expected-warning{{'array' is an unsafe buffer that does not perform bounds}}
};

void testRefersArrayStructFieldDecl(int i) {
  Struct2::array[i/2];     // expected-note{{used in buffer access here}}
}
}

int * return_ptr();

void testNoDeclRef(int i) {
  return_ptr() + i;   // expected-warning{{unsafe pointer arithmetic}}
  return_ptr()[i];    // expected-warning{{unsafe buffer access}}
}
