// RUN: %clang_cc1 -std=c++14 -Wno-unused-value %s -triple x86_64-linux-gnu -emit-llvm -o /dev/null

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
    a = a % b;                                            \
    a &b;                                                 \
    a | b;                                                \
    a ^ b;                                                \
    return a;                                             \
  }

MathShiftOps(FourCharsVecSize);
MathShiftOps(FourIntsVecSize);
MathShiftOps(FourLongLongsVecSize);
MathShiftOps(FourFloatsVecSize);
MathShiftOps(FourDoublesVecSize);
MathShiftOps(FourCharsExtVec);
MathShiftOps(FourIntsExtVec);
MathShiftOps(FourLongLongsExtVec);
MathShiftOps(FourFloatsExtVec);
MathShiftOps(FourDoublesExtVec);

MathShiftOpsInts(FourCharsVecSize);
MathShiftOpsInts(FourIntsVecSize);
MathShiftOpsInts(FourLongLongsVecSize);
MathShiftOpsInts(FourCharsExtVec);
MathShiftOpsInts(FourIntsExtVec);
MathShiftOpsInts(FourLongLongsExtVec);

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

// Only int vs float makes a difference here, so we only need to test 1 of each.
// Test Char to make sure the mixed-nature of shifts around char is evident.
void CharUsage() {
  constexpr auto a = FourCharsVecSize{6, 3, 2, 1} +
                     FourCharsVecSize{12, 15, 5, 7};
  static_assert(a[0] == 18 , "");
  static_assert(a[1] == 18 , "");
  static_assert(a[2] == 7 , "");
  static_assert(a[3] == 8 , "");

  constexpr auto b = FourCharsVecSize{19, 15, 13, 12} -
                     FourCharsVecSize{13, 14, 5, 3};
  static_assert(b[0] == 6 , "");
  static_assert(b[1] == 1 , "");
  static_assert(b[2] == 8 , "");
  static_assert(b[3] == 9 , "");

  constexpr auto c = FourCharsVecSize{8, 4, 2, 1} *
                     FourCharsVecSize{3, 4, 5, 6};
  static_assert(c[0] == 24 , "");
  static_assert(c[1] == 16 , "");
  static_assert(c[2] == 10 , "");
  static_assert(c[3] == 6 , "");

  constexpr auto d = FourCharsVecSize{12, 12, 10, 10} /
                     FourCharsVecSize{6, 4, 5, 2};
  static_assert(d[0] == 2 , "");
  static_assert(d[1] == 3 , "");
  static_assert(d[2] == 2 , "");
  static_assert(d[3] == 5 , "");
  
  constexpr auto e = FourCharsVecSize{12, 12, 10, 10} %
                     FourCharsVecSize{6, 4, 4, 3};
  static_assert(e[0] == 0 , "");
  static_assert(e[1] == 0 , "");
  static_assert(e[2] == 2 , "");
  static_assert(e[3] == 1 , "");

  constexpr auto f = FourCharsVecSize{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9 , "");
  static_assert(f[1] == 6 , "");
  static_assert(f[2] == 5 , "");
  static_assert(f[3] == 4 , "");

  constexpr auto g = FourCharsVecSize{19, 15, 12, 10} - 3;
  static_assert(g[0] == 16 , "");
  static_assert(g[1] == 12 , "");
  static_assert(g[2] == 9 , "");
  static_assert(g[3] == 7 , "");

  constexpr auto h = FourCharsVecSize{8, 4, 2, 1} * 3;
  static_assert(h[0] == 24 , "");
  static_assert(h[1] == 12 , "");
  static_assert(h[2] == 6 , "");
  static_assert(h[3] == 3 , "");

  constexpr auto j = FourCharsVecSize{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4 , "");
  static_assert(j[1] == 5 , "");
  static_assert(j[2] == 6 , "");
  static_assert(j[3] == 7 , "");
  
  constexpr auto k = FourCharsVecSize{12, 17, 19, 22} % 3;
  static_assert(k[0] == 0 , "");
  static_assert(k[1] == 2 , "");
  static_assert(k[2] == 1 , "");
  static_assert(k[3] == 1 , "");

  constexpr auto l = 3 + FourCharsVecSize{6, 3, 2, 1};
  static_assert(f[0] == 9 , "");
  static_assert(f[1] == 6 , "");
  static_assert(f[2] == 5 , "");
  static_assert(f[3] == 4 , "");

  constexpr auto m = 20 - FourCharsVecSize{19, 15, 12, 10};
  static_assert(m[0] == 1 , "");
  static_assert(m[1] == 5 , "");
  static_assert(m[2] == 8 , "");
  static_assert(m[3] == 10 , "");

  constexpr auto n = 3 * FourCharsVecSize{8, 4, 2, 1};
  static_assert(n[0] == 24 , "");
  static_assert(n[1] == 12 , "");
  static_assert(n[2] == 6 , "");
  static_assert(n[3] == 3 , "");

  constexpr auto o = 100 / FourCharsVecSize{12, 15, 18, 21};
  static_assert(o[0] == 8 , "");
  static_assert(o[1] == 6 , "");
  static_assert(o[2] == 5 , "");
  static_assert(o[3] == 4 , "");

  constexpr auto p = 100 % FourCharsVecSize{12, 15, 18, 21};
  static_assert(p[0] == 4 , "");
  static_assert(p[1] == 10 , "");
  static_assert(p[2] == 10 , "");
  static_assert(p[3] == 16 , "");

  constexpr auto q = FourCharsVecSize{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  static_assert(q[0] == 12 , "");
  static_assert(q[1] == 6 , "");
  static_assert(q[2] == 8 , "");
  static_assert(q[3] == 4 , "");

  constexpr auto r = FourCharsVecSize{19, 15, 12, 10} >>
                     FourCharsVecSize{1, 1, 2, 2};
  static_assert(r[0] == 9 , "");
  static_assert(r[1] == 7 , "");
  static_assert(r[2] == 3 , "");
  static_assert(r[3] == 2 , "");

  constexpr auto s = FourCharsVecSize{6, 3, 5, 10} << 1;
  static_assert(s[0] == 12 , "");
  static_assert(s[1] == 6 , "");
  static_assert(s[2] == 10 , "");
  static_assert(s[3] == 20 , "");

  constexpr auto t = FourCharsVecSize{19, 15, 10, 20} >> 1;
  static_assert(t[0] == 9 , "");
  static_assert(t[1] == 7 , "");
  static_assert(t[2] == 5 , "");
  static_assert(t[3] == 10 , "");

  constexpr auto u = 12 << FourCharsVecSize{1, 2, 3, 3};
  static_assert(u[0] == 24 , "");
  static_assert(u[1] == 48 , "");
  static_assert(u[2] == 96 , "");
  static_assert(u[3] == 96 , "");

  constexpr auto v = 12 >> FourCharsVecSize{1, 2, 2, 1};
  static_assert(v[0] == 6 , "");
  static_assert(v[1] == 3 , "");
  static_assert(v[2] == 3 , "");
  static_assert(v[3] == 6 , "");

  constexpr auto w = FourCharsVecSize{1, 2, 3, 4} <
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(w[0] == -1 , "");
  static_assert(w[1] == -1 , "");
  static_assert(w[2] == 0 , "");
  static_assert(w[3] == 0 , "");

  constexpr auto x = FourCharsVecSize{1, 2, 3, 4} >
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(x[0] == 0 , "");
  static_assert(x[1] == 0 , "");
  static_assert(x[2] == -1 , "");
  static_assert(x[3] == -1 , "");

  constexpr auto y = FourCharsVecSize{1, 2, 3, 4} <=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(y[0] == -1 , "");
  static_assert(y[1] == -1 , "");
  static_assert(y[2] == -1 , "");
  static_assert(y[3] == 0 , "");

  constexpr auto z = FourCharsVecSize{1, 2, 3, 4} >=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(z[0] == 0 , "");
  static_assert(z[1] == 0 , "");
  static_assert(z[2] == -1 , "");
  static_assert(z[3] == -1 , "");

  constexpr auto A = FourCharsVecSize{1, 2, 3, 4} ==
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(A[0] == 0 , "");
  static_assert(A[1] == 0 , "");
  static_assert(A[2] == -1 , "");
  static_assert(A[3] == 0 , "");

  constexpr auto B = FourCharsVecSize{1, 2, 3, 4} !=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(B[0] == -1 , "");
  static_assert(B[1] == -1 , "");
  static_assert(B[2] == 0 , "");
  static_assert(B[3] == -1 , "");

  constexpr auto C = FourCharsVecSize{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 , "");
  static_assert(C[1] == -1 , "");
  static_assert(C[2] == 0 , "");
  static_assert(C[3] == 0 , "");

  constexpr auto D = FourCharsVecSize{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 , "");
  static_assert(D[1] == 0 , "");
  static_assert(D[2] == 0 , "");
  static_assert(D[3] == -1 , "");

  constexpr auto E = FourCharsVecSize{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 , "");
  static_assert(E[1] == -1 , "");
  static_assert(E[2] == -1 , "");
  static_assert(E[3] == 0 , "");

  constexpr auto F = FourCharsVecSize{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 , "");
  static_assert(F[1] == 0 , "");
  static_assert(F[2] == -1 , "");
  static_assert(F[3] == -1 , "");

  constexpr auto G = FourCharsVecSize{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 , "");
  static_assert(G[1] == 0 , "");
  static_assert(G[2] == -1 , "");
  static_assert(G[3] == 0 , "");

  constexpr auto H = FourCharsVecSize{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 , "");
  static_assert(H[1] == -1 , "");
  static_assert(H[2] == 0 , "");
  static_assert(H[3] == -1 , "");

  constexpr auto I = FourCharsVecSize{1, 2, 3, 4} &
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(I[0] == 0 , "");
  static_assert(I[1] == 2 , "");
  static_assert(I[2] == 2 , "");
  static_assert(I[3] == 0 , "");

  constexpr auto J = FourCharsVecSize{1, 2, 3, 4} ^
                     FourCharsVecSize { 4, 3, 2, 1 };
  static_assert(J[0] == 5 , "");
  static_assert(J[1] == 1 , "");
  static_assert(J[2] == 1 , "");
  static_assert(J[3] == 5 , "");

  constexpr auto K = FourCharsVecSize{1, 2, 3, 4} |
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(K[0] == 5 , "");
  static_assert(K[1] == 3 , "");
  static_assert(K[2] == 3 , "");
  static_assert(K[3] == 5 , "");

  constexpr auto L = FourCharsVecSize{1, 2, 3, 4} & 3;
  static_assert(L[0] == 1 , "");
  static_assert(L[1] == 2 , "");
  static_assert(L[2] == 3 , "");
  static_assert(L[3] == 0 , "");

  constexpr auto M = FourCharsVecSize{1, 2, 3, 4} ^ 3;
  static_assert(M[0] == 2 , "");
  static_assert(M[1] == 1 , "");
  static_assert(M[2] == 0 , "");
  static_assert(M[3] == 7 , "");

  constexpr auto N = FourCharsVecSize{1, 2, 3, 4} | 3;
  static_assert(N[0] == 3 , "");
  static_assert(N[1] == 3 , "");
  static_assert(N[2] == 3 , "");
  static_assert(N[3] == 7 , "");

  constexpr auto O = FourCharsVecSize{5, 0, 6, 0} &&
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(O[0] == 1 , "");
  static_assert(O[1] == 0 , "");
  static_assert(O[2] == 0 , "");
  static_assert(O[3] == 0 , "");

  constexpr auto P = FourCharsVecSize{5, 0, 6, 0} ||
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(P[0] == 1 , "");
  static_assert(P[1] == 1 , "");
  static_assert(P[2] == 1 , "");
  static_assert(P[3] == 0 , "");

  constexpr auto Q = FourCharsVecSize{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 , "");
  static_assert(Q[1] == 0 , "");
  static_assert(Q[2] == 1 , "");
  static_assert(Q[3] == 0 , "");

  constexpr auto R = FourCharsVecSize{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 , "");
  static_assert(R[1] == 1 , "");
  static_assert(R[2] == 1 , "");
  static_assert(R[3] == 1 , "");

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 108 , "");
  static_assert(T[1] == 18 , "");
  static_assert(T[2] == 56 , "");
  static_assert(T[3] == 72 , "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(U[0] == 3 , "");
  static_assert(U[1] == 18 , "");
  static_assert(U[2] == 0 , "");
  static_assert(U[3] == 0 , "");

  constexpr auto V = CmpRem(a, b);
  static_assert(V[0] == 0 , "");
  static_assert(V[1] == 0 , "");
  static_assert(V[2] == 7 , "");
  static_assert(V[3] == 8 , "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 24 , "");
  static_assert(X[1] == 19 , "");
  static_assert(X[2] == 15 , "");
  static_assert(X[3] == 17 , "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 12 , "");
  static_assert(Y[1] == 17 , "");
  static_assert(Y[2] == -1 , "");
  static_assert(Y[3] == -1 , "");

  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 , "");
  static_assert(InvH[1] == 1 , "");
  static_assert(InvH[2] == 0 , "");
  static_assert(InvH[3] == 1 , "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(Z[0] == 36 , "");
  static_assert(Z[1] == 36 , "");
  static_assert(Z[2] == 7 , "");
  static_assert(Z[3] == 16 , "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(aa[0] == 9 , "");
  static_assert(aa[1] == 9 , "");
  static_assert(aa[2] == 7 , "");
  static_assert(aa[3] == 4 , "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(ab[0] == 2 , "");
  static_assert(ab[1] == 0 , "");
  static_assert(ab[2] == 0 , "");
  static_assert(ab[3] == 8 , "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(ac[0] == 20 , "");
  static_assert(ac[1] == 19 , "");
  static_assert(ac[2] == 15 , "");
  static_assert(ac[3] == 1 , "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(ad[0] == 22 , "");
  static_assert(ad[1] == 19 , "");
  static_assert(ad[2] == 15 , "");
  static_assert(ad[3] == 9 , "");

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  static_assert(ae[0] == -2 , "");
  static_assert(ae[1] == -3 , "");
  static_assert(ae[2] == -11 , "");
  static_assert(ae[3] == -21 , "");

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  static_assert(af[0] == -1 , "");
  static_assert(af[1] == 0 , "");
  static_assert(af[2] == 0 , "");
  static_assert(af[3] == 0 , "");
}

void CharExtVecUsage() {
  constexpr auto a = FourCharsExtVec{6, 3, 2, 1} +
                     FourCharsExtVec{12, 15, 5, 7};
  static_assert(a[0] == 18 , "");
  static_assert(a[1] == 18 , "");
  static_assert(a[2] == 7 , "");
  static_assert(a[3] == 8 , "");

  constexpr auto b = FourCharsExtVec{19, 15, 13, 12} -
                     FourCharsExtVec{13, 14, 5, 3};
  static_assert(b[0] == 6 , "");
  static_assert(b[1] == 1 , "");
  static_assert(b[2] == 8 , "");
  static_assert(b[3] == 9 , "");

  constexpr auto c = FourCharsExtVec{8, 4, 2, 1} *
                     FourCharsExtVec{3, 4, 5, 6};
  static_assert(c[0] == 24 , "");
  static_assert(c[1] == 16 , "");
  static_assert(c[2] == 10 , "");
  static_assert(c[3] == 6 , "");

  constexpr auto d = FourCharsExtVec{12, 12, 10, 10} /
                     FourCharsExtVec{6, 4, 5, 2};
  static_assert(d[0] == 2 , "");
  static_assert(d[1] == 3 , "");
  static_assert(d[2] == 2 , "");
  static_assert(d[3] == 5 , "");

  constexpr auto e = FourCharsExtVec{12, 12, 10, 10} %
                     FourCharsExtVec{6, 4, 4, 3};
  static_assert(e[0] == 0 , "");
  static_assert(e[1] == 0 , "");
  static_assert(e[2] == 2 , "");
  static_assert(e[3] == 1 , "");

  constexpr auto f = FourCharsExtVec{6, 3, 2, 1} + 3;
  static_assert(f[0] == 9 , "");
  static_assert(f[1] == 6 , "");
  static_assert(f[2] == 5 , "");
  static_assert(f[3] == 4 , "");

  constexpr auto g = FourCharsExtVec{19, 15, 12, 10} - 3;
  static_assert(g[0] == 16 , "");
  static_assert(g[1] == 12 , "");
  static_assert(g[2] == 9 , "");
  static_assert(g[3] == 7 , "");

  constexpr auto h = FourCharsExtVec{8, 4, 2, 1} * 3;
  static_assert(h[0] == 24 , "");
  static_assert(h[1] == 12 , "");
  static_assert(h[2] == 6 , "");
  static_assert(h[3] == 3 , "");

  constexpr auto j = FourCharsExtVec{12, 15, 18, 21} / 3;
  static_assert(j[0] == 4 , "");
  static_assert(j[1] == 5 , "");
  static_assert(j[2] == 6 , "");
  static_assert(j[3] == 7 , "");

  constexpr auto k = FourCharsExtVec{12, 17, 19, 22} % 3;
  static_assert(k[0] == 0 , "");
  static_assert(k[1] == 2 , "");
  static_assert(k[2] == 1 , "");
  static_assert(k[3] == 1 , "");

  constexpr auto l = 3 + FourCharsExtVec{6, 3, 2, 1};
  static_assert(l[0] == 9 , "");
  static_assert(l[1] == 6 , "");
  static_assert(l[2] == 5 , "");
  static_assert(l[3] == 4 , "");

  constexpr auto m = 20 - FourCharsExtVec{19, 15, 12, 10};
  static_assert(m[0] == 1 , "");
  static_assert(m[1] == 5 , "");
  static_assert(m[2] == 8 , "");
  static_assert(m[3] == 10 , "");

  constexpr auto n = 3 * FourCharsExtVec{8, 4, 2, 1};
  static_assert(n[0] == 24 , "");
  static_assert(n[1] == 12 , "");
  static_assert(n[2] == 6 , "");
  static_assert(n[3] == 3 , "");

  constexpr auto o = 100 / FourCharsExtVec{12, 15, 18, 21};
  static_assert(o[0] == 8 , "");
  static_assert(o[1] == 6 , "");
  static_assert(o[2] == 5 , "");
  static_assert(o[3] == 4 , "");

  constexpr auto p = 100 % FourCharsExtVec{12, 15, 18, 21};
  static_assert(p[0] == 4 , "");
  static_assert(p[1] == 10 , "");
  static_assert(p[2] == 10 , "");
  static_assert(p[3] == 16 , "");

  constexpr auto q = FourCharsExtVec{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  static_assert(q[0] == 12 , "");
  static_assert(q[1] == 6 , "");
  static_assert(q[2] == 8 , "");
  static_assert(q[3] == 4 , "");

  constexpr auto r = FourCharsExtVec{19, 15, 12, 10} >>
                     FourCharsExtVec{1, 1, 2, 2};
  static_assert(r[0] == 9 , "");
  static_assert(r[1] == 7 , "");
  static_assert(r[2] == 3 , "");
  static_assert(r[3] == 2 , "");

  constexpr auto s = FourCharsExtVec{6, 3, 5, 10} << 1;
  static_assert(s[0] == 12 , "");
  static_assert(s[1] == 6 , "");
  static_assert(s[2] == 10 , "");
  static_assert(s[3] == 20 , "");

  constexpr auto t = FourCharsExtVec{19, 15, 10, 20} >> 1;
  static_assert(t[0] == 9 , "");
  static_assert(t[1] == 7 , "");
  static_assert(t[2] == 5 , "");
  static_assert(t[3] == 10 , "");

  constexpr auto u = 12 << FourCharsExtVec{1, 2, 3, 3};
  static_assert(u[0] == 24 , "");
  static_assert(u[1] == 48 , "");
  static_assert(u[2] == 96 , "");
  static_assert(u[3] == 96 , "");

  constexpr auto v = 12 >> FourCharsExtVec{1, 2, 2, 1};
  static_assert(v[0] == 6 , "");
  static_assert(v[1] == 3 , "");
  static_assert(v[2] == 3 , "");
  static_assert(v[3] == 6 , "");

  constexpr auto w = FourCharsExtVec{1, 2, 3, 4} <
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(w[0] == -1 , "");
  static_assert(w[1] == -1 , "");
  static_assert(w[2] == 0 , "");
  static_assert(w[3] == 0 , "");

  constexpr auto x = FourCharsExtVec{1, 2, 3, 4} >
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(x[0] == 0 , "");
  static_assert(x[1] == 0 , "");
  static_assert(x[2] == -1 , "");
  static_assert(x[3] == -1 , "");

  constexpr auto y = FourCharsExtVec{1, 2, 3, 4} <=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(y[0] == -1 , "");
  static_assert(y[1] == -1 , "");
  static_assert(y[2] == -1 , "");
  static_assert(y[3] == 0 , "");

  constexpr auto z = FourCharsExtVec{1, 2, 3, 4} >=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(z[0] == 0 , "");
  static_assert(z[1] == 0 , "");
  static_assert(z[2] == -1 , "");
  static_assert(z[3] == -1 , "");

  constexpr auto A = FourCharsExtVec{1, 2, 3, 4} ==
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(A[0] == 0 , "");
  static_assert(A[1] == 0 , "");
  static_assert(A[2] == -1 , "");
  static_assert(A[3] == 0 , "");

  constexpr auto B = FourCharsExtVec{1, 2, 3, 4} !=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(B[0] == -1 , "");
  static_assert(B[1] == -1 , "");
  static_assert(B[2] == 0 , "");
  static_assert(B[3] == -1 , "");

  constexpr auto C = FourCharsExtVec{1, 2, 3, 4} < 3;
  static_assert(C[0] == -1 , "");
  static_assert(C[1] == -1 , "");
  static_assert(C[2] == 0 , "");
  static_assert(C[3] == 0 , "");

  constexpr auto D = FourCharsExtVec{1, 2, 3, 4} > 3;
  static_assert(D[0] == 0 , "");
  static_assert(D[1] == 0 , "");
  static_assert(D[2] == 0 , "");
  static_assert(D[3] == -1 , "");

  constexpr auto E = FourCharsExtVec{1, 2, 3, 4} <= 3;
  static_assert(E[0] == -1 , "");
  static_assert(E[1] == -1 , "");
  static_assert(E[2] == -1 , "");
  static_assert(E[3] == 0 , "");

  constexpr auto F = FourCharsExtVec{1, 2, 3, 4} >= 3;
  static_assert(F[0] == 0 , "");
  static_assert(F[1] == 0 , "");
  static_assert(F[2] == -1 , "");
  static_assert(F[3] == -1 , "");

  constexpr auto G = FourCharsExtVec{1, 2, 3, 4} == 3;
  static_assert(G[0] == 0 , "");
  static_assert(G[1] == 0 , "");
  static_assert(G[2] == -1 , "");
  static_assert(G[3] == 0 , "");

  constexpr auto H = FourCharsExtVec{1, 2, 3, 4} != 3;
  static_assert(H[0] == -1 , "");
  static_assert(H[1] == -1 , "");
  static_assert(H[2] == 0 , "");
  static_assert(H[3] == -1 , "");

  constexpr auto I = FourCharsExtVec{1, 2, 3, 4} &
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(I[0] == 0 , "");
  static_assert(I[1] == 2 , "");
  static_assert(I[2] == 2 , "");
  static_assert(I[3] == 0 , "");

  constexpr auto J = FourCharsExtVec{1, 2, 3, 4} ^
                     FourCharsExtVec { 4, 3, 2, 1 };
  static_assert(J[0] == 5 , "");
  static_assert(J[1] == 1 , "");
  static_assert(J[2] == 1 , "");
  static_assert(J[3] == 5 , "");

  constexpr auto K = FourCharsExtVec{1, 2, 3, 4} |
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(K[0] == 5 , "");
  static_assert(K[1] == 3 , "");
  static_assert(K[2] == 3 , "");
  static_assert(K[3] == 5 , "");

  constexpr auto L = FourCharsExtVec{1, 2, 3, 4} & 3;
  static_assert(L[0] == 1 , "");
  static_assert(L[1] == 2 , "");
  static_assert(L[2] == 3 , "");
  static_assert(L[3] == 0 , "");

  constexpr auto M = FourCharsExtVec{1, 2, 3, 4} ^ 3;
  static_assert(M[0] == 2 , "");
  static_assert(M[1] == 1 , "");
  static_assert(M[2] == 0 , "");
  static_assert(M[3] == 7 , "");

  constexpr auto N = FourCharsExtVec{1, 2, 3, 4} | 3;
  static_assert(N[0] == 3 , "");
  static_assert(N[1] == 3 , "");
  static_assert(N[2] == 3 , "");
  static_assert(N[3] == 7 , "");

  constexpr auto O = FourCharsExtVec{5, 0, 6, 0} &&
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(O[0] == 1 , "");
  static_assert(O[1] == 0 , "");
  static_assert(O[2] == 0 , "");
  static_assert(O[3] == 0 , "");

  constexpr auto P = FourCharsExtVec{5, 0, 6, 0} ||
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(P[0] == 1 , "");
  static_assert(P[1] == 1 , "");
  static_assert(P[2] == 1 , "");
  static_assert(P[3] == 0 , "");

  constexpr auto Q = FourCharsExtVec{5, 0, 6, 0} && 3;
  static_assert(Q[0] == 1 , "");
  static_assert(Q[1] == 0 , "");
  static_assert(Q[2] == 1 , "");
  static_assert(Q[3] == 0 , "");

  constexpr auto R = FourCharsExtVec{5, 0, 6, 0} || 3;
  static_assert(R[0] == 1 , "");
  static_assert(R[1] == 1 , "");
  static_assert(R[2] == 1 , "");
  static_assert(R[3] == 1 , "");

  constexpr auto T = CmpMul(a, b);
  static_assert(T[0] == 108 , "");
  static_assert(T[1] == 18 , "");
  static_assert(T[2] == 56 , "");
  static_assert(T[3] == 72 , "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(U[0] == 3 , "");
  static_assert(U[1] == 18 , "");
  static_assert(U[2] == 0 , "");
  static_assert(U[3] == 0 , "");

  constexpr auto V = CmpRem(a, b);
  static_assert(V[0] == 0 , "");
  static_assert(V[1] == 0 , "");
  static_assert(V[2] == 7 , "");
  static_assert(V[3] == 8 , "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(X[0] == 24 , "");
  static_assert(X[1] == 19 , "");
  static_assert(X[2] == 15 , "");
  static_assert(X[3] == 17 , "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(Y[0] == 12 , "");
  static_assert(Y[1] == 17 , "");
  static_assert(Y[2] == -1 , "");
  static_assert(Y[3] == -1 , "");

  constexpr auto InvH = -H;
  static_assert(InvH[0] == 1 , "");
  static_assert(InvH[1] == 1 , "");
  static_assert(InvH[2] == 0 , "");
  static_assert(InvH[3] == 1 , "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(Z[0] == 36 , "");
  static_assert(Z[1] == 36 , "");
  static_assert(Z[2] == 7 , "");
  static_assert(Z[3] == 16 , "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(aa[0] == 9 , "");
  static_assert(aa[1] == 9 , "");
  static_assert(aa[2] == 7 , "");
  static_assert(aa[3] == 4 , "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(ab[0] == 2 , "");
  static_assert(ab[1] == 0 , "");
  static_assert(ab[2] == 0 , "");
  static_assert(ab[3] == 8 , "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(ac[0] == 20 , "");
  static_assert(ac[1] == 19 , "");
  static_assert(ac[2] == 15 , "");
  static_assert(ac[3] == 1 , "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(ad[0] == 22 , "");
  static_assert(ad[1] == 19 , "");
  static_assert(ad[2] == 15 , "");
  static_assert(ad[3] == 9 , "");

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  static_assert(ae[0] == -2 , "");
  static_assert(ae[1] == -3 , "");
  static_assert(ae[2] == -11 , "");
  static_assert(ae[3] == -21 , "");

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  static_assert(af[0] == -1 , "");
  static_assert(af[1] == 0 , "");
  static_assert(af[2] == 0 , "");
  static_assert(af[3] == 0 , "");
}

void FloatUsage() {
  constexpr auto a = FourFloatsVecSize{6, 3, 2, 1} +
                     FourFloatsVecSize{12, 15, 5, 7};
  // CHECK: <4 x float> <float 1.800000e+01, float 1.800000e+01, float 7.000000e+00, float 8.000000e+00>
  constexpr auto b = FourFloatsVecSize{19, 15, 13, 12} -
                     FourFloatsVecSize{13, 14, 5, 3};
  // CHECK: store <4 x float> <float 6.000000e+00, float 1.000000e+00, float 8.000000e+00, float 9.000000e+00>
  constexpr auto c = FourFloatsVecSize{8, 4, 2, 1} *
                     FourFloatsVecSize{3, 4, 5, 6};
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.600000e+01, float 1.000000e+01, float 6.000000e+00>
  constexpr auto d = FourFloatsVecSize{12, 12, 10, 10} /
                     FourFloatsVecSize{6, 4, 5, 2};
  // CHECK: store <4 x float> <float 2.000000e+00, float 3.000000e+00, float 2.000000e+00, float 5.000000e+00>

  constexpr auto f = FourFloatsVecSize{6, 3, 2, 1} + 3;
  // CHECK: store <4 x float> <float 9.000000e+00, float 6.000000e+00, float 5.000000e+00, float 4.000000e+00>
  constexpr auto g = FourFloatsVecSize{19, 15, 12, 10} - 3;
  // CHECK: store <4 x float> <float 1.600000e+01, float 1.200000e+01, float 9.000000e+00, float 7.000000e+00>
  constexpr auto h = FourFloatsVecSize{8, 4, 2, 1} * 3;
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.200000e+01, float 6.000000e+00, float 3.000000e+00>
  constexpr auto j = FourFloatsVecSize{12, 15, 18, 21} / 3;
  // CHECK: store <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>

  constexpr auto l = 3 + FourFloatsVecSize{6, 3, 2, 1};
  // CHECK: store <4 x float> <float 9.000000e+00, float 6.000000e+00, float 5.000000e+00, float 4.000000e+00>
  constexpr auto m = 20 - FourFloatsVecSize{19, 15, 12, 10};
  // CHECK: store <4 x float> <float 1.000000e+00, float 5.000000e+00, float 8.000000e+00, float 1.000000e+01>
  constexpr auto n = 3 * FourFloatsVecSize{8, 4, 2, 1};
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.200000e+01, float 6.000000e+00, float 3.000000e+00>
  constexpr auto o = 100 / FourFloatsVecSize{12, 15, 18, 21};
  // CHECK: store <4 x float> <float 0x4020AAAAA0000000, float 0x401AAAAAA0000000, float 0x401638E380000000, float 0x40130C30C0000000>

  constexpr auto w = FourFloatsVecSize{1, 2, 3, 4} <
                     FourFloatsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 0>
  constexpr auto x = FourFloatsVecSize{1, 2, 3, 4} >
                     FourFloatsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto y = FourFloatsVecSize{1, 2, 3, 4} <=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 0>
  constexpr auto z = FourFloatsVecSize{1, 2, 3, 4} >=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto A = FourFloatsVecSize{1, 2, 3, 4} ==
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 0>
  constexpr auto B = FourFloatsVecSize{1, 2, 3, 4} !=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 -1>

  constexpr auto C = FourFloatsVecSize{1, 2, 3, 4} < 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 0>
  constexpr auto D = FourFloatsVecSize{1, 2, 3, 4} > 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 0, i32 -1>
  constexpr auto E = FourFloatsVecSize{1, 2, 3, 4} <= 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 0>
  constexpr auto F = FourFloatsVecSize{1, 2, 3, 4} >= 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto G = FourFloatsVecSize{1, 2, 3, 4} == 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 0>
  constexpr auto H = FourFloatsVecSize{1, 2, 3, 4} != 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 -1>

  constexpr auto O = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  constexpr auto P = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 0>

  constexpr auto Q = FourFloatsVecSize{5, 0, 6, 0} && 3;
  // CHECK: store <4 x i32> <i32 1, i32 0, i32 1, i32 0>
  constexpr auto R = FourFloatsVecSize{5, 0, 6, 0} || 3;
  // CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 1>

  constexpr auto T = CmpMul(a, b);
  // CHECK: store <4 x float> <float 1.080000e+02, float 1.800000e+01, float 5.600000e+01, float 7.200000e+01>

  constexpr auto U = CmpDiv(a, b);
  // CHECK: store <4 x float> <float 3.000000e+00, float 1.800000e+01, float 8.750000e-01, float 0x3FEC71C720000000>

  constexpr auto X = CmpAdd(a, b);
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.900000e+01, float 1.500000e+01, float 1.700000e+01>

  constexpr auto Y = CmpSub(a, b);
  // CHECK: store <4 x float> <float 1.200000e+01, float 1.700000e+01, float -1.000000e+00, float -1.000000e+00>

  constexpr auto Z = -Y;
  // CHECK: store <4 x float> <float -1.200000e+01, float -1.700000e+01, float 1.000000e+00, float 1.000000e+00>

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  // CHECK: store <4 x i32> <i32 -1, i32 0, i32 0, i32 0>
}

void FloatVecUsage() {
  constexpr auto a = FourFloatsVecSize{6, 3, 2, 1} +
                     FourFloatsVecSize{12, 15, 5, 7};
  // CHECK: <4 x float> <float 1.800000e+01, float 1.800000e+01, float 7.000000e+00, float 8.000000e+00>
  constexpr auto b = FourFloatsVecSize{19, 15, 13, 12} -
                     FourFloatsVecSize{13, 14, 5, 3};
  // CHECK: store <4 x float> <float 6.000000e+00, float 1.000000e+00, float 8.000000e+00, float 9.000000e+00>
  constexpr auto c = FourFloatsVecSize{8, 4, 2, 1} *
                     FourFloatsVecSize{3, 4, 5, 6};
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.600000e+01, float 1.000000e+01, float 6.000000e+00>
  constexpr auto d = FourFloatsVecSize{12, 12, 10, 10} /
                     FourFloatsVecSize{6, 4, 5, 2};
  // CHECK: store <4 x float> <float 2.000000e+00, float 3.000000e+00, float 2.000000e+00, float 5.000000e+00>

  constexpr auto f = FourFloatsVecSize{6, 3, 2, 1} + 3;
  // CHECK: store <4 x float> <float 9.000000e+00, float 6.000000e+00, float 5.000000e+00, float 4.000000e+00>
  constexpr auto g = FourFloatsVecSize{19, 15, 12, 10} - 3;
  // CHECK: store <4 x float> <float 1.600000e+01, float 1.200000e+01, float 9.000000e+00, float 7.000000e+00>
  constexpr auto h = FourFloatsVecSize{8, 4, 2, 1} * 3;
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.200000e+01, float 6.000000e+00, float 3.000000e+00>
  constexpr auto j = FourFloatsVecSize{12, 15, 18, 21} / 3;
  // CHECK: store <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>

  constexpr auto l = 3 + FourFloatsVecSize{6, 3, 2, 1};
  // CHECK: store <4 x float> <float 9.000000e+00, float 6.000000e+00, float 5.000000e+00, float 4.000000e+00>
  constexpr auto m = 20 - FourFloatsVecSize{19, 15, 12, 10};
  // CHECK: store <4 x float> <float 1.000000e+00, float 5.000000e+00, float 8.000000e+00, float 1.000000e+01>
  constexpr auto n = 3 * FourFloatsVecSize{8, 4, 2, 1};
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.200000e+01, float 6.000000e+00, float 3.000000e+00>
  constexpr auto o = 100 / FourFloatsVecSize{12, 15, 18, 21};
  // CHECK: store <4 x float> <float 0x4020AAAAA0000000, float 0x401AAAAAA0000000, float 0x401638E380000000, float 0x40130C30C0000000>

  constexpr auto w = FourFloatsVecSize{1, 2, 3, 4} <
                     FourFloatsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 0>
  constexpr auto x = FourFloatsVecSize{1, 2, 3, 4} >
                     FourFloatsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto y = FourFloatsVecSize{1, 2, 3, 4} <=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 0>
  constexpr auto z = FourFloatsVecSize{1, 2, 3, 4} >=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto A = FourFloatsVecSize{1, 2, 3, 4} ==
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 0>
  constexpr auto B = FourFloatsVecSize{1, 2, 3, 4} !=
                     FourFloatsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 -1>

  constexpr auto C = FourFloatsVecSize{1, 2, 3, 4} < 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 0>
  constexpr auto D = FourFloatsVecSize{1, 2, 3, 4} > 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 0, i32 -1>
  constexpr auto E = FourFloatsVecSize{1, 2, 3, 4} <= 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 -1, i32 0>
  constexpr auto F = FourFloatsVecSize{1, 2, 3, 4} >= 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 -1>
  constexpr auto G = FourFloatsVecSize{1, 2, 3, 4} == 3;
  // CHECK: store <4 x i32> <i32 0, i32 0, i32 -1, i32 0>
  constexpr auto H = FourFloatsVecSize{1, 2, 3, 4} != 3;
  // CHECK: store <4 x i32> <i32 -1, i32 -1, i32 0, i32 -1>

  constexpr auto O = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  constexpr auto P = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 0>

  constexpr auto Q = FourFloatsVecSize{5, 0, 6, 0} && 3;
  // CHECK: store <4 x i32> <i32 1, i32 0, i32 1, i32 0>
  constexpr auto R = FourFloatsVecSize{5, 0, 6, 0} || 3;
  // CHECK: store <4 x i32> <i32 1, i32 1, i32 1, i32 1>

  constexpr auto T = CmpMul(a, b);
  // CHECK: store <4 x float> <float 1.080000e+02, float 1.800000e+01, float 5.600000e+01, float 7.200000e+01>

  constexpr auto U = CmpDiv(a, b);
  // CHECK: store <4 x float> <float 3.000000e+00, float 1.800000e+01, float 8.750000e-01, float 0x3FEC71C720000000>

  constexpr auto X = CmpAdd(a, b);
  // CHECK: store <4 x float> <float 2.400000e+01, float 1.900000e+01, float 1.500000e+01, float 1.700000e+01>

  constexpr auto Y = CmpSub(a, b);
  // CHECK: store <4 x float> <float 1.200000e+01, float 1.700000e+01, float -1.000000e+00, float -1.000000e+00>

  constexpr auto Z = -Y;
  // CHECK: store <4 x float> <float -1.200000e+01, float -1.700000e+01, float 1.000000e+00, float 1.000000e+00>

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  // CHECK: store <4 x i32> <i32 -1, i32 0, i32 0, i32 0>
}

void I128Usage() {
  constexpr auto a = FourI128VecSize{1, 2, 3, 4};
  static_assert(a[0] == 1 , "");
  static_assert(a[1] == 2 , "");
  static_assert(a[2] == 3 , "");
  static_assert(a[3] == 4 , "");

  constexpr auto b = a < 3;
  static_assert(b[0] == -1 , "");
  static_assert(b[1] == -1 , "");
  static_assert(b[2] == 0 , "");
  static_assert(b[3] == 0 , "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128VecSize{1, 2, 10, 20};
  static_assert(c[0] == -2 , "");
  static_assert(c[1] == -3 , "");
  static_assert(c[2] == -11 , "");
  static_assert(c[3] == -21 , "");

  constexpr auto d = !FourI128VecSize{0, 1, 8, -1};
  static_assert(d[0] == -1 , "");
  static_assert(d[1] == 0 , "");
  static_assert(d[2] == 0 , "");
  static_assert(d[3] == 0 , "");
}

void I128VecUsage() {
  constexpr auto a = FourI128ExtVec{1, 2, 3, 4};
  static_assert(a[0] == 1 , "");
  static_assert(a[1] == 2 , "");
  static_assert(a[2] == 3 , "");
  static_assert(a[3] == 4 , "");

  constexpr auto b = a < 3;
  static_assert(b[0] == -1 , "");
  static_assert(b[1] == -1 , "");
  static_assert(b[2] == 0 , "");
  static_assert(b[3] == 0 , "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128ExtVec{1, 2, 10, 20};
  static_assert(c[0] == -2 , "");
  static_assert(c[1] == -3 , "");
  static_assert(c[2] == -11 , "");
  static_assert(c[3] == -21 , "");

  constexpr auto d = !FourI128ExtVec{0, 1, 8, -1};
  static_assert(d[0] == -1 , "");
  static_assert(d[1] == 0 , "");
  static_assert(d[2] == 0 , "");
  static_assert(d[3] == 0 , "");
}

using FourBoolsExtVec __attribute__((ext_vector_type(4))) = bool;
void BoolVecUsage() {
  constexpr auto a = FourBoolsExtVec{true, false, true, false} <
                     FourBoolsExtVec{false, false, true, true};
  static_assert(a[0] == false , "");
  static_assert(a[1] == false , "");
  static_assert(a[2] == false , "");
  static_assert(a[3] == true , "");

  constexpr auto b = FourBoolsExtVec{true, false, true, false} <=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(b[0] == false , "");
  static_assert(b[1] == true , "");
  static_assert(b[2] == true , "");
  static_assert(b[3] == true , "");

  constexpr auto c = FourBoolsExtVec{true, false, true, false} ==
                     FourBoolsExtVec{false, false, true, true};
  static_assert(c[0] == false , "");
  static_assert(c[1] == true , "");
  static_assert(c[2] == true , "");
  static_assert(c[3] == false , "");

  constexpr auto d = FourBoolsExtVec{true, false, true, false} !=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(d[0] == true , "");
  static_assert(d[1] == false , "");
  static_assert(d[2] == false , "");
  static_assert(d[3] == true , "");

  constexpr auto e = FourBoolsExtVec{true, false, true, false} >=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(e[0] == true , "");
  static_assert(e[1] == true , "");
  static_assert(e[2] == true , "");
  static_assert(e[3] == false , "");

  constexpr auto f = FourBoolsExtVec{true, false, true, false} >
                     FourBoolsExtVec{false, false, true, true};
  static_assert(f[0] == true , "");
  static_assert(f[1] == false , "");
  static_assert(f[2] == false , "");
  static_assert(f[3] == false , "");

  constexpr auto g = FourBoolsExtVec{true, false, true, false} &
                     FourBoolsExtVec{false, false, true, true};
  static_assert(g[0] == false , "");
  static_assert(g[1] == false , "");
  static_assert(g[2] == true , "");
  static_assert(g[3] == false , "");

  constexpr auto h = FourBoolsExtVec{true, false, true, false} |
                     FourBoolsExtVec{false, false, true, true};
  static_assert(h[0] == true , "");
  static_assert(h[1] == false , "");
  static_assert(h[2] == true , "");
  static_assert(h[3] == true , "");

  constexpr auto i = FourBoolsExtVec{true, false, true, false} ^
                     FourBoolsExtVec { false, false, true, true };
  static_assert(i[0] == true , "");
  static_assert(i[1] == false , "");
  static_assert(i[2] == false , "");
  static_assert(i[3] == true , "");

  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  static_assert(j[0] == false , "");
  static_assert(j[1] == true , "");
  static_assert(j[2] == false , "");
  static_assert(j[3] == true , "");

  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  static_assert(k[0] == false , "");
  static_assert(k[1] == true , "");
  static_assert(k[2] == false , "");
  static_assert(k[3] == true , "");
}