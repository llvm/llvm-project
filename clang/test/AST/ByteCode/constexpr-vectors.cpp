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
  constexpr auto w = FourCharsVecSize{1, 2, 3, 4} <
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(w[0] == -1 && w[1] == -1 && w[2] == 0 && w[3] == 0, "");

  constexpr auto x = FourCharsVecSize{1, 2, 3, 4} >
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(x[0] == 0 && x[1] == 0 && x[2] == -1 && x[3] == -1, "");

  constexpr auto y = FourCharsVecSize{1, 2, 3, 4} <=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(y[0] == -1 && y[1] == -1 && y[2] == -1 && y[3] == 0, "");

  constexpr auto z = FourCharsVecSize{1, 2, 3, 4} >=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(z[0] == 0 && z[1] == 0 && z[2] == -1 && z[3] == -1, "");

  constexpr auto A = FourCharsVecSize{1, 2, 3, 4} ==
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(A[0] == 0 && A[1] == 0 && A[2] == -1 && A[3] == 0, "");

  constexpr auto B = FourCharsVecSize{1, 2, 3, 4} !=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(B[0] == -1 && B[1] == -1 && B[2] == 0 && B[3] == -1, "");

  constexpr auto C = FourCharsVecSize{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 && C[1] == -1 && C[2] == 0 && C[3] == 0, "");

  constexpr auto D = FourCharsVecSize{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 && D[1] == 0 && D[2] == 0 && D[3] == -1, "");

  constexpr auto E = FourCharsVecSize{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 && E[1] == -1 && E[2] == -1 && E[3] == 0, "");

  constexpr auto F = FourCharsVecSize{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 && F[1] == 0 && F[2] == -1 && F[3] == -1, "");

  constexpr auto G = FourCharsVecSize{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 && G[1] == 0 && G[2] == -1 && G[3] == 0, "");

  constexpr auto H = FourCharsVecSize{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 && H[1] == -1 && H[2] == 0 && H[3] == -1, "");

  constexpr auto O = FourCharsVecSize{5, 0, 6, 0} &&
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(O[0] == 1 && O[1] == 0 && O[2] == 0 && O[3] == 0, "");

  constexpr auto P = FourCharsVecSize{5, 0, 6, 0} ||
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(P[0] == 1 && P[1] == 1 && P[2] == 1 && P[3] == 0, "");

  constexpr auto Q = FourCharsVecSize{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourCharsVecSize{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");

  constexpr auto H1 = FourCharsVecSize{-1, -1, 0, -1};
  constexpr auto InvH = -H1;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void CharExtVecUsage() {
  constexpr auto w = FourCharsExtVec{1, 2, 3, 4} <
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(w[0] == -1 && w[1] == -1 && w[2] == 0 && w[3] == 0, "");

  constexpr auto x = FourCharsExtVec{1, 2, 3, 4} >
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(x[0] == 0 && x[1] == 0 && x[2] == -1 && x[3] == -1, "");

  constexpr auto y = FourCharsExtVec{1, 2, 3, 4} <=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(y[0] == -1 && y[1] == -1 && y[2] == -1 && y[3] == 0, "");

  constexpr auto z = FourCharsExtVec{1, 2, 3, 4} >=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(z[0] == 0 && z[1] == 0 && z[2] == -1 && z[3] == -1, "");

  constexpr auto A = FourCharsExtVec{1, 2, 3, 4} ==
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(A[0] == 0 && A[1] == 0 && A[2] == -1 && A[3] == 0, "");

  constexpr auto B = FourCharsExtVec{1, 2, 3, 4} !=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(B[0] == -1 && B[1] == -1 && B[2] == 0 && B[3] == -1, "");

  constexpr auto C = FourCharsExtVec{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 && C[1] == -1 && C[2] == 0 && C[3] == 0, "");

  constexpr auto D = FourCharsExtVec{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 && D[1] == 0 && D[2] == 0 && D[3] == -1, "");

  constexpr auto E = FourCharsExtVec{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 && E[1] == -1 && E[2] == -1 && E[3] == 0, "");

  constexpr auto F = FourCharsExtVec{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 && F[1] == 0 && F[2] == -1 && F[3] == -1, "");

  constexpr auto G = FourCharsExtVec{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 && G[1] == 0 && G[2] == -1 && G[3] == 0, "");

  constexpr auto H = FourCharsExtVec{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 && H[1] == -1 && H[2] == 0 && H[3] == -1, "");

  constexpr auto O = FourCharsExtVec{5, 0, 6, 0} &&
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(O[0] == 1 && O[1] == 0 && O[2] == 0 && O[3] == 0, "");

  constexpr auto P = FourCharsExtVec{5, 0, 6, 0} ||
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(P[0] == 1 && P[1] == 1 && P[2] == 1 && P[3] == 0, "");

  constexpr auto Q = FourCharsExtVec{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourCharsExtVec{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");


  constexpr auto H1 = FourCharsExtVec{-1, -1, 0, -1};
  constexpr auto InvH = -H1;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatUsage() {
  constexpr auto w = FourFloatsVecSize{1, 2, 3, 4} <
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(w[0] == -1 && w[1] == -1 && w[2] == 0 && w[3] == 0, "");

  constexpr auto x = FourFloatsVecSize{1, 2, 3, 4} >
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(x[0] == 0 && x[1] == 0 && x[2] == -1 && x[3] == -1, "");

  constexpr auto y = FourFloatsVecSize{1, 2, 3, 4} <=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(y[0] == -1 && y[1] == -1 && y[2] == -1 && y[3] == 0, "");

  constexpr auto z = FourFloatsVecSize{1, 2, 3, 4} >=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(z[0] == 0 && z[1] == 0 && z[2] == -1 && z[3] == -1, "");

  constexpr auto A = FourFloatsVecSize{1, 2, 3, 4} ==
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(A[0] == 0 && A[1] == 0 && A[2] == -1 && A[3] == 0, "");

  constexpr auto B = FourFloatsVecSize{1, 2, 3, 4} !=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(B[0] == -1 && B[1] == -1 && B[2] == 0 && B[3] == -1, "");

  constexpr auto C = FourFloatsVecSize{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 && C[1] == -1 && C[2] == 0 && C[3] == 0, "");

  constexpr auto D = FourFloatsVecSize{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 && D[1] == 0 && D[2] == 0 && D[3] == -1, "");

  constexpr auto E = FourFloatsVecSize{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 && E[1] == -1 && E[2] == -1 && E[3] == 0, "");

  constexpr auto F = FourFloatsVecSize{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 && F[1] == 0 && F[2] == -1 && F[3] == -1, "");

  constexpr auto G = FourFloatsVecSize{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 && G[1] == 0 && G[2] == -1 && G[3] == 0, "");

  constexpr auto H = FourFloatsVecSize{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 && H[1] == -1 && H[2] == 0 && H[3] == -1, "");

  constexpr auto O1 = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(O1[0] == 1 && O1[1] == 0 && O1[2] == 0 && O1[3] == 0, "");

  constexpr auto P1 = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(P1[0] == 1 && P1[1] == 1 && P1[2] == 1 && P1[3] == 0, "");

  constexpr auto Q = FourFloatsVecSize{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourFloatsVecSize{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");


  constexpr auto Y = FourFloatsVecSize{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00};
  constexpr auto Z = -Y;
  static_assert(Z[0] == -1.200000e+01 && Z[1] == -1.700000e+01 && Z[2] == 1.000000e+00 && Z[3] == 1.000000e+00, "");

  constexpr auto O = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(O[0] == 1 && O[1] == 0 && O[2] == 0 && O[3] == 0, "");

  constexpr auto P = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(P[0] == 1 && P[1] == 1 && P[2] == 1 && P[3] == 0, "");

  // Operator ~ is illegal on floats.
  constexpr auto ae = ~FourFloatsVecSize{0, 1, 8, -1}; // expected-error {{invalid argument type}}

  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatVecUsage() {
  constexpr auto w = FourFloatsVecSize{1, 2, 3, 4} <
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(w[0] == -1 && w[1] == -1 && w[2] == 0 && w[3] == 0, "");

  constexpr auto x = FourFloatsVecSize{1, 2, 3, 4} >
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(x[0] == 0 && x[1] == 0 && x[2] == -1 && x[2] == -1, "");

  constexpr auto y = FourFloatsVecSize{1, 2, 3, 4} <=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(y[0] == -1 && y[1] == -1 && y[2] == -1 && y[3] == 0, "");

  constexpr auto z = FourFloatsVecSize{1, 2, 3, 4} >=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(z[0] == 0 && z[1] == 0 && z[2] == -1 && z[3] == -1, "");

  constexpr auto A = FourFloatsVecSize{1, 2, 3, 4} ==
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(A[0] == 0 && A[1] == 0 && A[2] == -1 && A[3] == 0, "");

  constexpr auto B = FourFloatsVecSize{1, 2, 3, 4} !=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(B[0] == -1 && B[1] == -1 && B[2] == 0 && B[3] == -1, "");

  constexpr auto C = FourFloatsVecSize{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 && C[1] == -1 && C[2] == 0 && C[3] == 0, "");

  constexpr auto D = FourFloatsVecSize{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 && D[1] == 0 && D[2] == 0 && D[3] == -1, "");

  constexpr auto E = FourFloatsVecSize{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 && E[1] == -1 && E[2] == -1 && E[3] == 0, "");

  constexpr auto F = FourFloatsVecSize{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 && F[1] == 0 && F[2] == -1 && F[3] == -1, "");

  constexpr auto G = FourFloatsVecSize{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 && G[1] == 0 && G[2] == -1 && G[3] == 0, "");

  constexpr auto H = FourFloatsVecSize{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 && H[1] == -1 && H[2] == 0 && H[3] == -1, "");

  constexpr auto O = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(O[0] == 1 && O[1] == 0 && O[2] == 0 && O[3] == 0, "");

  constexpr auto P = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(P[0] == 1 && P[1] == 1 && P[2] == 1 && P[3] == 0, "");

  constexpr auto Q = FourFloatsVecSize{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourFloatsVecSize{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");

  constexpr auto Y = FourFloatsVecSize{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00};
  constexpr auto Z = -Y;
  static_assert(Z[0] == -1.200000e+01 && Z[1] == -1.700000e+01 && Z[2] == 1.000000e+00 && Z[3] == 1.000000e+00, "");

  // Operator ~ is illegal on floats.
  constexpr auto ae = ~FourFloatsVecSize{0, 1, 8, -1}; // expected-error {{invalid argument type}}

  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void I128Usage() {
  constexpr auto a = FourI128VecSize{1, 2, 3, 4};
  static_assert(a[0] == 1 && a[1] == 2 && a[2] == 3 && a[3] == 4, "");

  constexpr auto a1 = FourI128VecSize{5, 0, 6, 0} && FourI128VecSize{5, 5, 0, 0};
  static_assert(a1[0] == 1 && a1[1] == 0 && a1[2] == 0 && a1[3] == 0, "");

  constexpr auto a2 = FourI128VecSize{5, 0, 6, 0} || FourI128VecSize{5, 5, 0, 0};
  static_assert(a2[0] == 1 && a2[1] == 1 && a2[2] == 1 && a2[3] == 0, "");

  constexpr auto Q = FourI128VecSize{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourI128VecSize{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");

  constexpr auto b = a < 3;
  static_assert(b[0] == -1 && b[1] == -1 && b[2] == 0 && b[3] == 0, "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128VecSize{1, 2, 10, 20};
   static_assert(c[0] == -2 && c[1] == -3 && c[2] == -11 && c[3] == -21, "");

  constexpr auto d = !FourI128VecSize{0, 1, 8, -1};
  static_assert(d[0] == -1 && d[1] == 0 && d[2] == 0 && d[3] == 0, "");
}

void I128VecUsage() {
  constexpr auto a = FourI128ExtVec{1, 2, 3, 4};
  static_assert(a[0] == 1 && a[1] == 2 && a[2] == 3 && a[3] == 4, "");

  constexpr auto a1 = FourI128ExtVec{5, 0, 6, 0} && FourI128ExtVec{5, 5, 0, 0};
  static_assert(a1[0] == 1 && a1[1] == 0 && a1[2] == 0 && a1[3] == 0, "");

  constexpr auto a2 = FourI128ExtVec{5, 0, 6, 0} || FourI128ExtVec{5, 5, 0, 0};
  static_assert(a2[0] == 1 && a2[1] == 1 && a2[2] == 1 && a2[3] == 0, "");

  constexpr auto Q = FourI128ExtVec{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 && Q[1] == 0 && Q[2] == 1 && Q[3] == 0, "");

  constexpr auto R = FourI128ExtVec{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 && R[1] == 1 && R[2] == 1 && R[3] == 1, "");

  constexpr auto b = a < 3;
  static_assert(b[0] == -1 && b[1] == -1 && b[2] == 0 && b[3] == 0, "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128ExtVec{1, 2, 10, 20};
  static_assert(c[0] == -2 && c[1] == -3 && c[2] == -11 && c[3] == -21, "");

  constexpr auto d = !FourI128ExtVec{0, 1, 8, -1};
  static_assert(d[0] == -1 && d[1] == 0 && d[2] == 0 && d[3] == 0, "");
}

using FourBoolsExtVec __attribute__((ext_vector_type(4))) = bool;
void BoolVecUsage() {
  constexpr auto a = FourBoolsExtVec{true, false, true, false} <
                     FourBoolsExtVec{false, false, true, true};
  static_assert(a[0] == false && a[1] == false && a[2] == false && a[3] == true, "");

  constexpr auto b = FourBoolsExtVec{true, false, true, false} <=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(b[0] == false && b[1] == true && b[2] == true && b[3] == true, "");

  constexpr auto c = FourBoolsExtVec{true, false, true, false} ==
                     FourBoolsExtVec{false, false, true, true};
  static_assert(c[0] == false && c[1] == true && c[2] == true && c[3] == false, "");

  constexpr auto d = FourBoolsExtVec{true, false, true, false} !=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(d[0] == true && d[1] == false && d[2] == false && d[3] == true, "");

  constexpr auto e = FourBoolsExtVec{true, false, true, false} >=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(e[0] == true && e[1] == true && e[2] == true && e[3] == false, "");

  constexpr auto f = FourBoolsExtVec{true, false, true, false} >
                     FourBoolsExtVec{false, false, true, true};
  static_assert(f[0] == true && f[1] == false && f[2] == false && f[3] == false, "");

  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  static_assert(j[0] == false && j[1] == true && j[2] == false && j[3] == true, "");

  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  static_assert(k[0] == false && k[1] == true && k[2] == false && k[3] == true, "");
}
