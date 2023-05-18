// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc -verify %s -Wbitfield-enum-conversion
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux -verify %s -Wbitfield-enum-conversion

enum TwoBits { Hi1 = 3 } two_bits;
enum TwoBitsSigned { Lo2 = -2, Hi2 = 1 } two_bits_signed;
enum ThreeBits { Hi3 = 7 } three_bits;
enum ThreeBitsSigned { Lo4 = -4, Hi4 = 3 } three_bits_signed;
enum TwoBitsFixed : unsigned { Hi5 = 3 } two_bits_fixed;

struct Foo {
  unsigned two_bits : 2;        // expected-note 2 {{widen this field to 3 bits}} expected-note 2 {{type signed}}
  int two_bits_signed : 2;      // expected-note 2 {{widen this field to 3 bits}} expected-note 1 {{type unsigned}}
  unsigned three_bits : 3;      // expected-note 2 {{type signed}}
  int three_bits_signed : 3;    // expected-note 1 {{type unsigned}}

#ifdef _WIN32
  // expected-note@+2 {{type unsigned}}
#endif
  ThreeBits three_bits_enum : 3;
  ThreeBits four_bits_enum : 4;
};

void f() {
  Foo f;

  f.two_bits = two_bits;
  f.two_bits = two_bits_signed;            // expected-warning {{negative enumerators}}
  f.two_bits = three_bits;                 // expected-warning {{not wide enough}}
  f.two_bits = three_bits_signed;          // expected-warning {{negative enumerators}} expected-warning {{not wide enough}}
  f.two_bits = two_bits_fixed;

  f.two_bits_signed = two_bits;            // expected-warning {{needs an extra bit}}
  f.two_bits_signed = two_bits_signed;
  f.two_bits_signed = three_bits;          // expected-warning {{not wide enough}}
  f.two_bits_signed = three_bits_signed;   // expected-warning {{not wide enough}}

  f.three_bits = two_bits;
  f.three_bits = two_bits_signed;          // expected-warning {{negative enumerators}}
  f.three_bits = three_bits;
  f.three_bits = three_bits_signed;        // expected-warning {{negative enumerators}}

  f.three_bits_signed = two_bits;
  f.three_bits_signed = two_bits_signed;
  f.three_bits_signed = three_bits;        // expected-warning {{needs an extra bit}}
  f.three_bits_signed = three_bits_signed;

#ifdef _WIN32
  // Enums on Windows are always implicitly 'int', which is signed, so you need
  // an extra bit to store values that set the MSB. This is not true on SysV
  // platforms like Linux.
  // expected-warning@+2 {{needs an extra bit}}
#endif
  f.three_bits_enum = three_bits;
  f.four_bits_enum = three_bits;

  // Explicit casts suppress the warning.
  f.two_bits = (unsigned)three_bits_signed;
  f.two_bits = static_cast<unsigned>(three_bits_signed);
}
