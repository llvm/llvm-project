// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t -verify %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -fsyntax-only -verify %s

#ifndef HEADER
#define HEADER

enum class [[clang::flag_enum]] FlagEnum {
  A = 0x1,
  B = 0x2,
  C = 0x4
};
// expected-warning@-5 {{'operator&' is not available for flag-like enumeration type 'FlagEnum'}} \
// expected-warning@-5 {{'operator^' is not available for flag-like enumeration type 'FlagEnum'}} \
// expected-warning@-5 {{'operator~' is not available for flag-like enumeration type 'FlagEnum'}}

constexpr FlagEnum operator|(FlagEnum lhs, FlagEnum rhs) {
  return static_cast<FlagEnum>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

#else

int f(FlagEnum flags) {
  switch (flags) {
  case FlagEnum::A:
    return 1;
  case FlagEnum::B | FlagEnum::C: // no-warning
    return 2;
  case FlagEnum(8): // expected-warning {{case value not in enumerated type 'FlagEnum'}}
    return 3;
  default:
    return 0;
  }
}

#endif
