// Check that an enum with the flag_enum attribute that is deserialized from a
// PCH does not crash Sema, and that its flag bits are still computed correctly.

// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s

#ifndef HEADER
#define HEADER

enum __attribute__((flag_enum)) FlagEnum {
  A = 0x1,
  B = 0x2,
  C = 0x4,
};

#else

// The flag bits of FlagEnum are only known to Sema if they were recomputed
// after deserialization, so a wrong or missing cache entry shows up here as
// either a crash or a bogus diagnostic on the first two cases.
int f(enum FlagEnum e) {
  switch (e) {
  case A:
    return 1;
  case B | C: // no-warning
    return 2;
  case 0x8: // expected-warning {{case value not in enumerated type 'enum FlagEnum'}}
    return 3;
  default:
    return 0;
  }
}

#endif
