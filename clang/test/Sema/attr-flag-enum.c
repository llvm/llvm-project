// RUN: %clang_cc1 -triple %itanium_abi_triple -verify -fsyntax-only -std=c11 -Wassign-enum %s

enum __attribute__((flag_enum)) flag {
  ea = 0x1,
  eb = 0x2,
  ec = 0x8,
};

enum __attribute__((flag_enum)) {
  g = 0x7,  // expected-warning {{enumeration value 'g' is out of range of flags in enumeration type 'enum (unnamed at}}
};

enum __attribute__((flag_enum)) flag2 {
  ga = 0x1,
  gb = 0x4,

  gc = 0x5, // no-warning
  gd = 0x7, // expected-warning {{enumeration value 'gd' is out of range}}
  ge = ~0x2, // expected-warning {{enumeration value 'ge' is out of range}}
  gf = ~0x4, // no-warning
  gg = ~0x1, // no-warning
  gh = ~0x5, // no-warning
  gi = ~0x11, // expected-warning {{enumeration value 'gi' is out of range}}
};

enum __attribute__((flag_enum)) flag3 {
  fa = 0x1,
  fb = ~0x1u, // no-warning
};

// What happens here is that ~0x2 is negative, and so the enum must be signed.
// But ~0x1u is unsigned and has the high bit set, so the enum must be 64-bit.
// The result is that ~0x1u does not have high bits set, and so it is considered
// to be an invalid value. See Sema::IsValueInFlagEnum in SemaDecl.cpp for more
// discussion.
enum __attribute__((flag_enum)) flag4 {
  ha = 0x1,
  hb = 0x2,

  hc = ~0x1u, // expected-warning {{enumeration value 'hc' is out of range}}
  hd = ~0x2, // no-warning
};

void f(void) {
  enum flag e = 0; // no-warning
  e = 0x1; // no-warning
  e = 0x3; // no-warning
  e = 0xa; // no-warning
  e = 0x4; // expected-warning {{integer constant not in range of enumerated type}}
  e = 0xf; // expected-warning {{integer constant not in range of enumerated type}}
  e = ~0; // no-warning
  e = ~0x1; // no-warning
  e = ~0x2; // no-warning
  e = ~0x3; // no-warning
  e = ~0x4; // expected-warning {{integer constant not in range of enumerated type}}

  switch (e) {
    case 0: break; // no-warning
    case 0x1: break; // no-warning
    case 0x3: break; // no-warning
    case 0xa: break; // no-warning
    case 0x4: break; // expected-warning {{case value not in enumerated type}}
    case 0xf: break; // expected-warning {{case value not in enumerated type}}
    case ~0: break; // expected-warning {{case value not in enumerated type}}
    case ~0x1: break; // expected-warning {{case value not in enumerated type}}
    case ~0x2: break; // expected-warning {{case value not in enumerated type}}
    case ~0x3: break; // expected-warning {{case value not in enumerated type}}
    case ~0x4: break; // expected-warning {{case value not in enumerated type}}
    default: break;
  }

  enum flag2 f = ~0x1; // no-warning
  f = ~0x1u; // no-warning

  enum flag4 h = ~0x1; // no-warning
  h = ~0x1u; // expected-warning {{integer constant not in range of enumerated type}}
}
