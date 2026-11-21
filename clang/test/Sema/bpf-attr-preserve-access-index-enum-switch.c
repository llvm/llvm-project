// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,reloc -Wswitch %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -verify=expected,reloc -Wswitch %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=expected,noreloc -Wswitch \
// RUN:   -fbpf-no-enum-reloc=E %s

// A relocated enumerator in a case label is not diagnosed by
// -Wbpf-enum-not-relocated. A relocated case value is only known at load time,
// so it is not checked for duplicates or overlaps, and a constant switch
// condition may match it. The other checks use the local values, as with
// -fbpf-no-enum-reloc=.

enum E { A = 1, B = 4, C = 8, D = 300 } __attribute__((preserve_access_index));

int f(int i, char c, enum E e) {
  switch (i) {
  case A: // noreloc-note {{previous case defined here}}
    return 1;
  case 1: // noreloc-error {{duplicate case value: 'A' and '1' both equal '1'}}
    return 2;
  case B: // noreloc-note {{previous case defined here}}
    return 3;
  case B + 1 ... C: // noreloc-error {{duplicate case value '6'}}
    return 4;
  case B: // noreloc-error {{duplicate case value 'B'}}
    return 5;
  case C + 1 ... B: // expected-warning {{empty case range specified}}
    return 6;
  case 6: // noreloc-note {{previous case defined here}}
    return 7;
  case A + __builtin_preserve_enum_value(*(enum E *)B, 1): // expected-error {{expression is not an integer constant expression}} expected-note {{an explicit '__builtin_preserve_enum_value' call is relocated at load time, so it is not a constant}} expected-note {{a case label with an enumerator of an enum with the 'preserve_access_index' attribute is relocated, and compared like an 'if' condition before the switch}}
    return 8;
  // A compound literal is not an integer rvalue, so A is kept a constant.
  case (int){A + 10}: // reloc-error {{enumerator 'A' is not relocated where a constant is required, although 'enum E' has the 'preserve_access_index' attribute}}
    return 9;
  }

  // The cases without a relocated value are still checked.
  switch (i) {
  case 2: // expected-note {{previous case defined here}}
  case B - 2: // noreloc-error {{duplicate case value '2'}} noreloc-note {{previous case defined here}}
  case 2: // expected-error {{duplicate case value '2'}}
  case 20 ... 22: // expected-error {{duplicate case value '21'}}
  case 21: // expected-note {{previous case defined here}}
    break;
  }

  switch (2) { // noreloc-warning {{no case matching constant switch condition '2'}}
  case A:
    break;
  }

  switch (e) { // expected-warning {{enumeration value 'D' not handled in switch}}
  case A:
  case B ... C:
    break;
  }

  // A relocated range may overlap another range. 4 is B, 10 is not in enum E.
  switch (e) { // reloc-warning {{enumeration value 'D' not handled in switch}}
  case A ... C: // noreloc-note {{previous case defined here}}
  case 4 ... 10: // reloc-warning {{case value not in enumerated type 'enum E'}} noreloc-error {{duplicate case value '8'}}
    break;
  }

  switch (c) {
  case D: // expected-warning {{overflow converting case value to switch condition type (300 to 44)}}
    break;
  }
  return 0;
}
