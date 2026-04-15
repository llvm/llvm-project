// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=expected,ref %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=expected,newinterp %s -fexperimental-new-constant-interpreter

// Test for C++26 erroneous behavior in constant expressions (P2795R5)
// Reading an uninitialized/erroneous value in a constant expression is an error.

// Direct read of default-initialized variable
constexpr int test1() {
  int x;        // default-initialized, has erroneous value
  return x;     // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val1 = test1();  // expected-error {{constexpr variable 'val1' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test1()'}}

// Reading member with erroneous value
struct S {
  int x;
  constexpr S() {}  // x has erroneous value
};

constexpr int test2() {
  S s;
  return s.x;   // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val2 = test2();  // expected-error {{constexpr variable 'val2' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test2()'}}

// [[indeterminate]] in constexpr - also an error
constexpr int test3() {
  [[indeterminate]] int x;  // x has indeterminate value (UB in general, error in constexpr)
  return x;                 // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val3 = test3();  // expected-error {{constexpr variable 'val3' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test3()'}}

// Proper initialization is fine
constexpr int test4() {
  int x = 42;
  return x;
}
constexpr int val4 = test4();  // OK

// Array with erroneous elements
constexpr int test5() {
  int arr[3];  // elements have erroneous values
  return arr[0];  // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val5 = test5();  // expected-error {{constexpr variable 'val5' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test5()'}}

// Partial initialization - uninitialized portion is erroneous
constexpr int test6() {
  int arr[3] = {1};  // arr[1] and arr[2] are zero-initialized, not erroneous
  return arr[1];     // OK - zero-initialized
}
constexpr int val6 = test6();  // OK, val6 == 0

// (P2795R5 [bit.cast]) Erroneous/indeterminate values should propagate through bit_cast.

// bit_cast of erroneous value to non-byte type is an error.
// The erroneous bytes become indeterminate in BitCastBuffer, so the diagnostic
// reports them as indeterminate. This is the current behavior; a fully
// conforming implementation would distinguish erroneous from indeterminate
// per [bit.cast]/2.
constexpr int test_bitcast_erroneous() {
  int x;  // erroneous value
  return __builtin_bit_cast(int, x); // ref-note {{indeterminate value can only initialize an object of type 'unsigned char'}} \
                                     // newinterp-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val_bc1 = test_bitcast_erroneous(); // expected-error {{constexpr variable 'val_bc1' must be initialized by a constant expression}} \
                                                   // expected-note {{in call to 'test_bitcast_erroneous()'}}

// bit_cast of [[indeterminate]] value to non-byte type is an error
constexpr int test_bitcast_indeterminate() {
  [[indeterminate]] int x;  // indeterminate value
  return __builtin_bit_cast(int, x); // ref-note {{indeterminate value can only initialize an object of type 'unsigned char'}} \
                                     // newinterp-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val_bc2 = test_bitcast_indeterminate(); // expected-error {{constexpr variable 'val_bc2' must be initialized by a constant expression}} \
                                                       // expected-note {{in call to 'test_bitcast_indeterminate()'}}

// bit_cast of erroneous value to unsigned char preserves the uninitialized
// status. 
// 
// Per P2795R5 [bit.cast]/2, the result has erroneous value for
// unsigned char / std::byte. Reading it in constexpr is still an error.
constexpr unsigned char test_bitcast_erroneous_to_byte() {
  unsigned char x;  // erroneous value
  unsigned char y = __builtin_bit_cast(unsigned char, x); // newinterp-note {{read of uninitialized object is not allowed in a constant expression}}
  return y; // ref-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr unsigned char val_bc3 = test_bitcast_erroneous_to_byte(); // expected-error {{constexpr variable 'val_bc3' must be initialized by a constant expression}} \
                                                                     // expected-note {{in call to 'test_bitcast_erroneous_to_byte()'}}

// bit_cast of properly initialized value is fine
constexpr int test_bitcast_ok() {
  int x = 42;
  return __builtin_bit_cast(int, x);
}
constexpr int val_bc4 = test_bitcast_ok();  // OK

// bit_cast of erroneous struct member
struct BitCastSrc {
  int a;
  constexpr BitCastSrc() {} // a has erroneous value
};

struct BitCastDst {
  int a;
};

constexpr int test_bitcast_struct_erroneous() {
  BitCastSrc src;
  BitCastDst dst = __builtin_bit_cast(BitCastDst, src); // expected-note {{indeterminate value can only initialize an object of type 'unsigned char'}}
  return dst.a;
}
constexpr int val_bc5 = test_bitcast_struct_erroneous(); // expected-error {{constexpr variable 'val_bc5' must be initialized by a constant expression}} \
                                                          // expected-note {{in call to 'test_bitcast_struct_erroneous()'}}
