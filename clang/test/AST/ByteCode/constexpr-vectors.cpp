// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++14 -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter -std=c++14 -fsyntax-only -verify

using FourCharsVecSize __attribute__((vector_size(4))) = char;
using FourIntsVecSize __attribute__((vector_size(16))) = int;
using FourLongLongsVecSize __attribute__((vector_size(32))) = long long;
using FourFloatsVecSize __attribute__((vector_size(16))) = float;
using FourDoublesVecSize __attribute__((vector_size(32))) = double;
using FourI128VecSize __attribute__((vector_size(64))) = __int128;

using FourCharsExtVec __attribute__((ext_vector_type(4))) = char;
using FourIntsExtVec __attribute__((ext_vector_type(4))) = int;
using FourI128ExtVec __attribute__((ext_vector_type(4))) = __int128;

// Only int vs float makes a difference here, so we only need to test 1 of each.
// Test Char to make sure the mixed-nature of shifts around char is evident.
void CharUsage() {
  constexpr auto H = FourCharsVecSize{-1, -1, 0, -1};
  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void CharExtVecUsage() {
  constexpr auto H = FourCharsExtVec{-1, -1, 0, -1};
  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatUsage() {
  constexpr auto Y = FourFloatsVecSize{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00};
  constexpr auto Z = -Y;
  static_assert(Z[0] == -1.200000e+01 && Z[1] == -1.700000e+01 && Z[2] == 1.000000e+00 && Z[3] == 1.000000e+00, "");

  // Operator ~ is illegal on floats.
  constexpr auto ae = ~FourFloatsVecSize{0, 1, 8, -1}; // expected-error {{invalid argument type}}

  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatVecUsage() {
  constexpr auto Y = FourFloatsVecSize{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00};
  constexpr auto Z = -Y;
  static_assert(Z[0] == -1.200000e+01 && Z[1] == -1.700000e+01 && Z[2] == 1.000000e+00 && Z[3] == 1.000000e+00, "");

  // Operator ~ is illegal on floats.
  constexpr auto ae = ~FourFloatsVecSize{0, 1, 8, -1}; // expected-error {{invalid argument type}}

  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void I128Usage() {
  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128VecSize{1, 2, 10, 20};
   static_assert(c[0] == -2 && c[1] == -3 && c[2] == -11 && c[3] == -21, "");

  constexpr auto d = !FourI128VecSize{0, 1, 8, -1};
  static_assert(d[0] == -1 && d[1] == 0 && d[2] == 0 && d[3] == 0, "");
}

void I128VecUsage() {
  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128ExtVec{1, 2, 10, 20};
  static_assert(c[0] == -2 && c[1] == -3 && c[2] == -11 && c[3] == -21, "");

  constexpr auto d = !FourI128ExtVec{0, 1, 8, -1};
  static_assert(d[0] == -1 && d[1] == 0 && d[2] == 0 && d[3] == 0, "");
}

using FourBoolsExtVec __attribute__((ext_vector_type(4))) = bool;
void BoolVecUsage() {
  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  static_assert(j[0] == false && j[1] == true && j[2] == false && j[3] == true, "");

  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  static_assert(k[0] == false && k[1] == true && k[2] == false && k[3] == true, "");
}
