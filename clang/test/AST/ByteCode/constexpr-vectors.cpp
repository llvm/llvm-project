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
using FourLongLongsExtVec __attribute__((ext_vector_type(4))) = long long;
using FourFloatsExtVec __attribute__((ext_vector_type(4))) = float;
using FourDoublesExtVec __attribute__((ext_vector_type(4))) = double;
using FourI128ExtVec __attribute__((ext_vector_type(4))) = __int128;

// Next a series of tests to make sure these operations are usable in
// constexpr functions. Template instantiations don't emit Winvalid-constexpr,
// so we have to do these as macros.
#define MathShiftOps(Type)                            \
  constexpr auto MathShiftOps##Type(Type a, Type b) { \
    a = a + b;                                        \
    a = a - b;                                        \
    a = a * b;                                        \
    a = a / b;                                        \
    b = a + 1;                                        \
    b = a - 1;                                        \
    b = a * 1;                                        \
    b = a / 1;                                        \
    a += a;                                           \
    a -= a;                                           \
    a *= a;                                           \
    a /= a;                                           \
    b += a;                                           \
    b -= a;                                           \
    b *= a;                                           \
    b /= a;                                           \
    b = (a += a);                                     \
    b = (a -= a);                                     \
    b = (a *= a);                                     \
    b = (a /= a);                                     \
    b = (b += a);                                     \
    b = (b -= a);                                     \
    b = (b *= a);                                     \
    b = (b /= a);                                     \
    a < b;                                            \
    a > b;                                            \
    a <= b;                                           \
    a >= b;                                           \
    a == b;                                           \
    a != b;                                           \
    a &&b;                                            \
    a || b;                                           \
    auto c = (a, b);                                  \
    return c;                                         \
  }

// Ops specific to Integers.
#define MathShiftOpsInts(Type)                            \
  constexpr auto MathShiftopsInts##Type(Type a, Type b) { \
    a = a << b;                                           \
    a = a >> b;                                           \
    a = a << 3;                                           \
    a = a >> 3;                                           \
    a = 3 << b;                                           \
    a = 3 >> b;                                           \
    a <<= b;                                              \
    a >>= b;                                              \
    a <<= 3;                                              \
    a >>= 3;                                              \
    b = (a <<= b);                                        \
    b = (a >>= b);                                        \
    b = (a <<= 3);                                        \
    b = (a >>= 3);                                        \
    a = a % b;                                            \
    a &b;                                                 \
    a | b;                                                \
    a ^ b;                                                \
    return a;                                             \
  }

MathShiftOps(FourCharsVecSize);
MathShiftOps(FourIntsVecSize);
MathShiftOps(FourLongLongsVecSize);
MathShiftOps(FourI128VecSize);
MathShiftOps(FourFloatsVecSize);
MathShiftOps(FourDoublesVecSize);
MathShiftOps(FourCharsExtVec);
MathShiftOps(FourIntsExtVec);
MathShiftOps(FourLongLongsExtVec);
MathShiftOps(FourI128ExtVec);
MathShiftOps(FourFloatsExtVec);
MathShiftOps(FourDoublesExtVec);

MathShiftOpsInts(FourCharsVecSize);
MathShiftOpsInts(FourIntsVecSize);
MathShiftOpsInts(FourLongLongsVecSize);
MathShiftOpsInts(FourI128VecSize);
MathShiftOpsInts(FourCharsExtVec);
MathShiftOpsInts(FourIntsExtVec);
MathShiftOpsInts(FourLongLongsExtVec);
MathShiftOpsInts(FourI128ExtVec);

template <typename T, typename U>
constexpr auto CmpMul(T t, U u) {
  t *= u;
  return t;
}
template <typename T, typename U>
constexpr auto CmpDiv(T t, U u) {
  t /= u;
  return t;
}
template <typename T, typename U>
constexpr auto CmpRem(T t, U u) {
  t %= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpAdd(T t, U u) {
  t += u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpSub(T t, U u) {
  t -= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpLSH(T t, U u) {
  t <<= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpRSH(T t, U u) {
  t >>= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinAnd(T t, U u) {
  t &= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinXOr(T t, U u) {
  t ^= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinOr(T t, U u) {
  t |= u;
  return t;
}

constexpr auto CmpF(float t, float u) {
  return __builtin_fabs(t - u) < 0.0001;
}

// Only int vs float makes a difference here, so we only need to test 1 of each.
// Test Char to make sure the mixed-nature of shifts around char is evident.
void CharUsage() {
  constexpr auto a = FourCharsVecSize{6, 3, 2, 1} +
            FourCharsVecSize{12, 15, 5, 7};
  static_assert(a[0] == 18 && a[1] == 18 && a[2] == 7 && a[3] == 8, "");

  constexpr auto b = FourCharsVecSize{19, 15, 13, 12} -
                     FourCharsVecSize{13, 14, 5, 3};
  static_assert(b[0] == 6 && b[1] == 1 && b[2] == 8 && b[3] == 9, "");

  constexpr auto c = FourCharsVecSize{8, 4, 2, 1} *
                     FourCharsVecSize{3, 4, 5, 6};
  static_assert(c[0] == 24 && c[1] == 16 && c[2] == 10 && c[3] == 6, "");

  constexpr auto d = FourCharsVecSize{12, 12, 10, 10} /
                     FourCharsVecSize{6, 4, 5, 2};
  static_assert(d[0] == 2 && d[1] == 3 && d[2] == 2 && d[3] == 5, "");

  constexpr auto e = FourCharsVecSize{12, 12, 10, 10} %
                     FourCharsVecSize{6, 4, 4, 3};
  static_assert(e[0] == 0 && e[1] == 0 && e[2] == 2 && e[3] == 1, "");

  constexpr auto f = FourCharsVecSize{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9 && f[1] == 6 && f[2] == 5 && f[3] == 4, "");

  constexpr auto g = FourCharsVecSize{19, 15, 12, 10} - 3;
  static_assert(g[0] == 16 && g[1] == 12 && g[2] == 9 && g[3] == 7, "");

  constexpr auto h = FourCharsVecSize{8, 4, 2, 1} * 3;
  static_assert(h[0] == 24 && h[1] == 12 && h[2] == 6 && h[3] == 3, "");

  constexpr auto j = FourCharsVecSize{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4 && j[1] == 5 && j[2] == 6 && j[3] == 7, "");

  constexpr auto k = FourCharsVecSize{12, 17, 19, 22} % 3;
  static_assert(k[0] == 0 && k[1] == 2 && k[2] == 1 && k[3] == 1, "");

  constexpr auto l = 3 + FourCharsVecSize{6, 3, 2, 1};
  static_assert(l[0] == 9 && l[1] == 6 && l[2] == 5 && l[3] == 4, "");

  constexpr auto m = 20 - FourCharsVecSize{19, 15, 12, 10};
  static_assert(m[0] == 1 && m[1] == 5 && m[2] == 8 && m[3] == 10, "");

  constexpr auto n = 3 * FourCharsVecSize{8, 4, 2, 1};
  static_assert(n[0] == 24 && n[1] == 12 && n[2] == 6 && n[3] == 3, "");

  constexpr auto o = 100 / FourCharsVecSize{12, 15, 18, 21};
  static_assert(o[0] == 8 && o[1] == 6 && o[2] == 5 && o[3] == 4, "");

  constexpr auto p = 100 % FourCharsVecSize{12, 15, 18, 21};
  static_assert(p[0] == 4 && p[1] == 10 && p[2] == 10 && p[3] == 16, "");

  constexpr auto q = FourCharsVecSize{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  static_assert(q[0] == 12 && q[1] == 6 && q[2] == 8 && q[3] == 4, "");

  constexpr auto r = FourCharsVecSize{19, 15, 12, 10} >>
                     FourCharsVecSize{1, 1, 2, 2};
  static_assert(r[0] == 9 && r[1] == 7 && r[2] == 3 && r[3] == 2, "");

  constexpr auto s = FourCharsVecSize{6, 3, 5, 10} << 1;
  static_assert(s[0] == 12 && s[1] == 6 && s[2] == 10 && s[3] == 20, "");

  constexpr auto t = FourCharsVecSize{19, 15, 10, 20} >> 1;
  static_assert(t[0] == 9 && t[1] == 7 && t[2] == 5 && t[3] == 10, "");

  constexpr auto u = 12 << FourCharsVecSize{1, 2, 3, 3};
  static_assert(u[0] == 24 && u[1] == 48 && u[2] == 96 && u[3] == 96, "");

  constexpr auto v = 12 >> FourCharsVecSize{1, 2, 2, 1};
  static_assert(v[0] == 6 && v[1] == 3 && v[2] == 3 && v[3] == 6, "");

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

  constexpr auto I = FourCharsVecSize{1, 2, 3, 4} &
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(I[0] == 0 && I[1] == 2 && I[2] == 2 && I[3] == 0, "");

  constexpr auto J = FourCharsVecSize{1, 2, 3, 4} ^
                     FourCharsVecSize { 4, 3, 2, 1 };
  static_assert(J[0] == 5 && J[1] == 1 && J[2] == 1 && J[3] == 5, "");

  constexpr auto K = FourCharsVecSize{1, 2, 3, 4} |
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(K[0] == 5 && K[1] == 3 && K[2] == 3 && K[3] == 5, "");

  constexpr auto L = FourCharsVecSize{1, 2, 3, 4} & 3;
  static_assert(L[0] == 1 && L[1] == 2 && L[2] == 3 && L[3] == 0, "");

  constexpr auto M = FourCharsVecSize{1, 2, 3, 4} ^ 3;
  static_assert(M[0] == 2 && M[1] == 1 && M[2] == 0 && M[3] == 7, "");

  constexpr auto N = FourCharsVecSize{1, 2, 3, 4} | 3;
  static_assert(N[0] == 3 && N[1] == 3 && N[2] == 3 && N[3] == 7, "");

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

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 108 && T[1] == 18 && T[2] == 56 && T[3] == 72, "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(U[0] == 3 && U[1] == 18 && U[2] == 0 && U[3] == 0, "");

  constexpr auto V = CmpRem(a, b);
  static_assert(V[0] == 0 && V[1] == 0 && V[2] == 7 && V[3] == 8, "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 24 && X[1] == 19 && X[2] == 15 && X[3] == 17, "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 12 && Y[1] == 17 && Y[2] == -1 && Y[3] == -1, "");

  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(Z[0] == 36 && Z[1] == 36 && Z[2] == 7 && Z[3] == 16, "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(aa[0] == 9 && aa[1] == 9 && aa[2] == 7 && aa[3] == 4, "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(ab[0] == 2 && ab[1] == 0 && ab[2] == 0 && ab[3] == 8, "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(ac[0] == 20 && ac[1] == 19 && ac[2] == 15 && ac[3] == 1, "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(ad[0] == 22 && ad[1] == 19 && ad[2] == 15 && ad[3] == 9, "");

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void CharExtVecUsage() {
  constexpr auto a = FourCharsExtVec{6, 3, 2, 1} +
                     FourCharsExtVec{12, 15, 5, 7};
  static_assert(a[0] == 18 && a[1] == 18 && a[2] == 7 && a[3] == 8, "");

  constexpr auto b = FourCharsExtVec{19, 15, 13, 12} -
                     FourCharsExtVec{13, 14, 5, 3};
  static_assert(b[0] == 6 && b[1] == 1 && b[2] == 8 && b[3] == 9, "");

  constexpr auto c = FourCharsExtVec{8, 4, 2, 1} *
                     FourCharsExtVec{3, 4, 5, 6};
  static_assert(c[0] == 24 && c[1] == 16 && c[2] == 10 && c[3] == 6, "");

  constexpr auto d = FourCharsExtVec{12, 12, 10, 10} /
                     FourCharsExtVec{6, 4, 5, 2};
  static_assert(d[0] == 2 && d[1] == 3 && d[2] == 2 && d[3] == 5, "");

  constexpr auto e = FourCharsExtVec{12, 12, 10, 10} %
                     FourCharsExtVec{6, 4, 4, 3};
  static_assert(e[0] == 0 && e[1] == 0 && e[2] == 2 && e[3] == 1, "");

  constexpr auto f = FourCharsExtVec{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9 && f[1] == 6 && f[2] == 5 && f[3] == 4, "");

  constexpr auto g = FourCharsExtVec{19, 15, 12, 10} - 3;
  static_assert(g[0] == 16 && g[1] == 12 && g[2] == 9 && g[3] == 7, "");

  constexpr auto h = FourCharsExtVec{8, 4, 2, 1} * 3;
  static_assert(h[0] == 24 && h[1] == 12 && h[2] == 6 && h[3] == 3, "");

  constexpr auto j = FourCharsExtVec{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4 && j[1] == 5 && j[2] == 6 && j[3] == 7, "");

  constexpr auto k = FourCharsExtVec{12, 17, 19, 22} % 3;
  static_assert(k[0] == 0 && k[1] == 2 && k[2] == 1 && k[3] == 1, "");

  constexpr auto l = 3 + FourCharsExtVec{6, 3, 2, 1};
  static_assert(l[0] == 9 && l[1] == 6 && l[2] == 5 && l[3] == 4, "");

  constexpr auto m = 20 - FourCharsExtVec{19, 15, 12, 10};
  static_assert(m[0] == 1 && m[1] == 5 && m[2] == 8 && m[3] == 10, "");

  constexpr auto n = 3 * FourCharsExtVec{8, 4, 2, 1};
  static_assert(n[0] == 24 && n[1] == 12 && n[2] == 6 && n[3] == 3, "");

  constexpr auto o = 100 / FourCharsExtVec{12, 15, 18, 21};
  static_assert(o[0] == 8 && o[1] == 6 && o[2] == 5 && o[3] == 4, "");

  constexpr auto p = 100 % FourCharsExtVec{12, 15, 18, 21};
  static_assert(p[0] == 4 && p[1] == 10 && p[2] == 10 && p[3] == 16, "");

  constexpr auto q = FourCharsExtVec{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  static_assert(q[0] == 12 && q[1] == 6 && q[2] == 8 && q[3] == 4, "");

  constexpr auto r = FourCharsExtVec{19, 15, 12, 10} >>
                     FourCharsExtVec{1, 1, 2, 2};
  static_assert(r[0] == 9 && r[1] == 7 && r[2] == 3 && r[3] == 2, "");

  constexpr auto s = FourCharsExtVec{6, 3, 5, 10} << 1;
  static_assert(s[0] == 12 && s[1] == 6 && s[2] == 10 && s[3] == 20, "");

  constexpr auto t = FourCharsExtVec{19, 15, 10, 20} >> 1;
  static_assert(t[0] == 9 && t[1] == 7 && t[2] == 5 && t[3] == 10, "");

  constexpr auto u = 12 << FourCharsExtVec{1, 2, 3, 3};
  static_assert(u[0] == 24 && u[1] == 48 && u[2] == 96 && u[3] == 96, "");

  constexpr auto v = 12 >> FourCharsExtVec{1, 2, 2, 1};
  static_assert(v[0] == 6 && v[1] == 3 && v[2] == 3 && v[3] == 6, "");

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

  constexpr auto I = FourCharsExtVec{1, 2, 3, 4} &
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(I[0] == 0 && I[1] == 2 && I[2] == 2 && I[3] == 0, "");

  constexpr auto J = FourCharsExtVec{1, 2, 3, 4} ^
                     FourCharsExtVec { 4, 3, 2, 1 };
  static_assert(J[0] == 5 && J[1] == 1 && J[2] == 1 && J[3] == 5, "");

  constexpr auto K = FourCharsExtVec{1, 2, 3, 4} |
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(K[0] == 5 && K[1] == 3 && K[2] == 3 && K[3] == 5, "");

  constexpr auto L = FourCharsExtVec{1, 2, 3, 4} & 3;
  static_assert(L[0] == 1 && L[1] == 2 && L[2] == 3 && L[3] == 0, "");

  constexpr auto M = FourCharsExtVec{1, 2, 3, 4} ^ 3;
  static_assert(M[0] == 2 && M[1] == 1 && M[2] == 0 && M[3] == 7, "");

  constexpr auto N = FourCharsExtVec{1, 2, 3, 4} | 3;
  static_assert(N[0] == 3 && N[1] == 3 && N[2] == 3 && N[3] == 7, "");

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

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 108 && T[1] == 18 && T[2] == 56 && T[3] == 72, "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(U[0] == 3 && U[1] == 18 && U[2] == 0 && U[3] == 0, "");

  constexpr auto V = CmpRem(a, b);
  static_assert(V[0] == 0 && V[1] == 0 && V[2] == 7 && V[3] == 8, "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 24 && X[1] == 19 && X[2] == 15 && X[3] == 17, "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 12 && Y[1] == 17 && Y[2] == -1 && Y[3] == -1, "");

  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 && InvH[1] == 1 && InvH[2] == 0 && InvH[3] == 1, "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(Z[0] == 36 && Z[1] == 36 && Z[2] == 7 && Z[3] == 16, "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(aa[0] == 9 && aa[1] == 9 && aa[2] == 7 && aa[3] == 4, "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(ab[0] == 2 && ab[1] == 0 && ab[2] == 0 && ab[3] == 8, "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(ac[0] == 20 && ac[1] == 19 && ac[2] == 15 && ac[3] == 1, "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(ad[0] == 22 && ad[1] == 19 && ad[2] == 15 && ad[3] == 9, "");

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  static_assert(ae[0] == -2 && ae[1] == -3 && ae[2] == -11 && ae[3] == -21, "");

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatUsage() {
  constexpr auto a = FourFloatsVecSize{6, 3, 2, 1} +
                     FourFloatsVecSize{12, 15, 5, 7};
  static_assert(a[0] == 1.800000e+01 && a[1] == 1.800000e+01 && a[2] == 7.000000e+00 && a[3] == 8.000000e+00, "");

  constexpr auto b = FourFloatsVecSize{19, 15, 13, 12} -
                     FourFloatsVecSize{13, 14, 5, 3};
  static_assert(b[0] == 6.000000e+00 && b[1] == 1.000000e+00 && b[2] == 8.000000e+00 && b[3] == 9.000000e+00, "");

  constexpr auto c = FourFloatsVecSize{8, 4, 2, 1} *
                     FourFloatsVecSize{3, 4, 5, 6};
  static_assert(c[0] == 2.400000e+01 && c[1] == 1.600000e+01 && c[2] == 1.000000e+01 && c[3] == 6.000000e+00, "");

  constexpr auto d = FourFloatsVecSize{12, 12, 10, 10} /
                     FourFloatsVecSize{6, 4, 5, 2};
  static_assert(d[0] == 2.000000e+00 && d[1] == 3.000000e+00 && d[2] == 2.000000e+00 && d[3] == 5.000000e+00, "");

  constexpr auto f = FourFloatsVecSize{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9.000000e+00 && f[1] == 6.000000e+00 && f[2] == 5.000000e+00 && f[3] == 4.000000e+00, "");

  constexpr auto g = FourFloatsVecSize{19, 15, 12, 10} - 3;
  static_assert(g[0] == 1.600000e+01 && g[1] == 1.200000e+01 && g[2] == 9.000000e+00 && g[3] == 7.000000e+00, "");

  constexpr auto h = FourFloatsVecSize{8, 4, 2, 1} * 3;
  static_assert(h[0] == 2.400000e+01 && h[1] == 1.200000e+01 && h[2] == 6.000000e+00 && h[3] == 3.000000e+00, "");

  constexpr auto j = FourFloatsVecSize{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4.000000e+00 && j[1] == 5.000000e+00 && j[2] == 6.000000e+00 && j[3] == 7.000000e+00, "");

  constexpr auto l = 3 + FourFloatsVecSize{6, 3, 2, 1};
  static_assert(l[0] == 9.000000e+00 && l[1] == 6.000000e+00 && l[2] == 5.000000e+00 && l[3] == 4.000000e+00, "");

  constexpr auto m = 20 - FourFloatsVecSize{19, 15, 12, 10};
  static_assert(m[0] == 1.000000e+00 && m[1] == 5.000000e+00 && m[2] == 8.000000e+00 && m[3] == 1.000000e+01, "");

  constexpr auto n = 3 * FourFloatsVecSize{8, 4, 2, 1};
  static_assert(n[0] == 2.400000e+01 && n[1] == 1.200000e+01 && n[2] == 6.000000e+00 && n[3] == 3.000000e+00, "");

  constexpr auto o = 100 / FourFloatsVecSize{12, 15, 18, 21};
  static_assert(CmpF(o[0], 100.0 / 12) && CmpF(o[1], 100.0 / 15) && CmpF(o[2], 100.0 / 18) && CmpF(o[3], 100.0 / 21), "");

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

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 1.080000e+02 && T[1] == 1.800000e+01 && T[2] == 5.600000e+01 && T[3] == 7.200000e+01, "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(CmpF(U[0], a[0] / b[0]) && CmpF(U[1], a[1] / b[1]) && CmpF(U[2], a[2] / b[2]) && CmpF(U[3], a[3] / b[3]), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 2.400000e+01 && X[1] == 1.900000e+01 && X[2] == 1.500000e+01 && X[3] == 1.700000e+01, "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 1.200000e+01 && Y[1] == 1.700000e+01 && Y[2] == -1.000000e+00 && Y[3] == -1.000000e+00, "");

  constexpr auto Z = -Y;
  static_assert(Z[0] == -1.200000e+01 && Z[1] == -1.700000e+01 && Z[2] == 1.000000e+00 && Z[3] == 1.000000e+00, "");

  // Operator ~ is illegal on floats.
  constexpr auto ae = ~FourFloatsVecSize{0, 1, 8, -1}; // expected-error {{invalid argument type}}

  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 && af[1] == 0 && af[2] == 0 && af[3] == 0, "");
}

void FloatVecUsage() {
  constexpr auto a = FourFloatsVecSize{6, 3, 2, 1} +
                     FourFloatsVecSize{12, 15, 5, 7};
  static_assert(a[0] == 1.800000e+01 && a[1] == 1.800000e+01 && a[2] == 7.000000e+00 && a[3] == 8.000000e+00, "");

  constexpr auto b = FourFloatsVecSize{19, 15, 13, 12} -
                     FourFloatsVecSize{13, 14, 5, 3};
  static_assert(b[0] == 6.000000e+00 && b[1] == 1.000000e+00 && b[2] == 8.000000e+00 && b[3] == 9.000000e+00, "");

  constexpr auto c = FourFloatsVecSize{8, 4, 2, 1} *
                     FourFloatsVecSize{3, 4, 5, 6};
  static_assert(c[0] == 2.400000e+01 && c[1] == 1.600000e+01 && c[2] == 1.000000e+01 && c[3] == 6.000000e+00, "");

  constexpr auto d = FourFloatsVecSize{12, 12, 10, 10} /
                     FourFloatsVecSize{6, 4, 5, 2};
  static_assert(d[0] == 2.000000e+00 && d[1] == 3.000000e+00 && d[2] == 2.000000e+00 && d[3] == 5.000000e+00, "");

  constexpr auto f = FourFloatsVecSize{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9.000000e+00 && f[1] == 6.000000e+00 && f[2] == 5.000000e+00 && f[3] == 4.000000e+00, "");

  constexpr auto g = FourFloatsVecSize{19, 15, 12, 10} - 3;
  static_assert(g[0] == 1.600000e+01 && g[1] == 1.200000e+01 && g[2] == 9.000000e+00 && g[3] == 7.000000e+00, "");

  constexpr auto h = FourFloatsVecSize{8, 4, 2, 1} * 3;
  static_assert(h[0] == 2.400000e+01 && h[1] == 1.200000e+01 && h[2] == 6.000000e+00 && h[3] == 3.000000e+00, "");

  constexpr auto j = FourFloatsVecSize{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4.000000e+00 && j[1] == 5.000000e+00 && j[2] == 6.000000e+00 && j[3] == 7.000000e+00, "");

  constexpr auto l = 3 + FourFloatsVecSize{6, 3, 2, 1};
  static_assert(l[0] == 9.000000e+00 && l[1] == 6.000000e+00 && l[2] == 5.000000e+00 && l[3] == 4.000000e+00, "");

  constexpr auto m = 20 - FourFloatsVecSize{19, 15, 12, 10};
  static_assert(m[0] == 1.000000e+00 && m[1] == 5.000000e+00 && m[2] == 8.000000e+00 && m[3] == 1.000000e+01, "");

  constexpr auto n = 3 * FourFloatsVecSize{8, 4, 2, 1};
  static_assert(n[0] == 2.400000e+01 && n[1] == 1.200000e+01 && n[2] == 6.000000e+00 && n[3] == 3.000000e+00, "");

  constexpr auto o = 100 / FourFloatsVecSize{12, 15, 18, 21};
  static_assert(CmpF(o[0], 100.0 / 12) && CmpF(o[1], 100.0 / 15) && CmpF(o[2], 100.0 / 18) && CmpF(o[3], 100.0 / 21), "");

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

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 1.080000e+02 && T[1] == 1.800000e+01 && T[2] == 5.600000e+01 && T[3] == 7.200000e+01, "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(CmpF(U[0], a[0] / b[0]) && CmpF(U[1], a[1] / b[1]) && CmpF(U[2], a[2] / b[2]) && CmpF(U[3], a[3] / b[3]), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 2.400000e+01 && X[1] == 1.900000e+01 && X[2] == 1.500000e+01 && X[3] == 1.700000e+01, "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 1.200000e+01 && Y[1] == 1.700000e+01 && Y[2] == -1.000000e+00 && Y[3] == -1.000000e+00, "");

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

  constexpr auto g = FourBoolsExtVec{true, false, true, false} &
                     FourBoolsExtVec{false, false, true, true};
  static_assert(g[0] == false && g[1] == false && g[2] == true && g[3] == false, "");

  constexpr auto h = FourBoolsExtVec{true, false, true, false} |
                     FourBoolsExtVec{false, false, true, true};
  static_assert(h[0] == true && h[1] == false && h[2] == true && h[3] == true, "");

  constexpr auto i = FourBoolsExtVec{true, false, true, false} ^
                     FourBoolsExtVec { false, false, true, true };
  static_assert(i[0] == true && i[1] == false && i[2] == false && i[3] == true, "");

  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  static_assert(j[0] == false && j[1] == true && j[2] == false && j[3] == true, "");

  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  static_assert(k[0] == false && k[1] == true && k[2] == false && k[3] == true, "");
}

using EightBoolsExtVec __attribute__((ext_vector_type(8))) = bool;
void BoolVecShuffle() {
  constexpr EightBoolsExtVec a = __builtin_shufflevector(
      FourBoolsExtVec{}, FourBoolsExtVec{}, 0, 1, 2, 3, 4, 5, 6, 7);
}
