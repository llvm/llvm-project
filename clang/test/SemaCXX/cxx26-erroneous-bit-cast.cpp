// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=expected,ref %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=expected,newinterp %s -fexperimental-new-constant-interpreter

// FIXME: Erroneous bytes are tracked as indeterminate in the bit_cast buffer.
constexpr int erroneous_to_int() {
  int x;                             // newinterp-note {{declared here}}
  return __builtin_bit_cast(int, x); // ref-note {{indeterminate value can only initialize an object of type 'unsigned char'}} \
                                     // newinterp-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v1 = erroneous_to_int(); // expected-error {{constexpr variable 'v1' must be initialized by a constant expression}} \
                                       // expected-note {{in call to 'erroneous_to_int()'}}

constexpr int indeterminate_to_int() {
  [[indeterminate]] int x;           // newinterp-note {{declared here}}
  return __builtin_bit_cast(int, x); // ref-note {{indeterminate value can only initialize an object of type 'unsigned char'}} \
                                     // newinterp-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v2 = indeterminate_to_int(); // expected-error {{constexpr variable 'v2' must be initialized by a constant expression}} \
                                           // expected-note {{in call to 'indeterminate_to_int()'}}

constexpr unsigned char erroneous_to_byte() {
  unsigned char x;                                        // newinterp-note {{declared here}}
  unsigned char y = __builtin_bit_cast(unsigned char, x); // newinterp-note {{read of uninitialized object is not allowed in a constant expression}} \
                                                          // ref-note {{declared here}}
  return y;                                               // ref-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr unsigned char v3 = erroneous_to_byte(); // expected-error {{constexpr variable 'v3' must be initialized by a constant expression}} \
                                                  // expected-note {{in call to 'erroneous_to_byte()'}}

constexpr int initialized() {
  int x = 42;
  return __builtin_bit_cast(int, x);
}
static_assert(initialized() == 42);

struct Src {
  int a;
  constexpr Src() {}
};
struct Dst {
  int a;
};
constexpr int erroneous_member() {
  Src src;
  Dst dst = __builtin_bit_cast(Dst, src); // expected-note {{indeterminate value can only initialize an object of type 'unsigned char'}}
  return dst.a;
}
constexpr int v5 = erroneous_member(); // expected-error {{constexpr variable 'v5' must be initialized by a constant expression}} \
                                       // expected-note {{in call to 'erroneous_member()'}}
