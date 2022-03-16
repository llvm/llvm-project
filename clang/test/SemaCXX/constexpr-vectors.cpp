// RUN: %clang_cc1 -std=c++14 -Wno-unused-value %s -disable-llvm-passes -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

// FIXME: Unfortunately there is no good way to validate that our values are
// correct since Vector types don't have operator [] implemented for constexpr.
// Instead, we need to use filecheck to ensure the emitted IR is correct. Once
// someone implements array subscript operator for these types as constexpr,
// this test should modified to jsut use static asserts.

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
  // CHECK: store <4 x i8> <i8 18, i8 18, i8 7, i8 8>
  constexpr auto b = FourCharsVecSize{19, 15, 13, 12} -
                     FourCharsVecSize{13, 14, 5, 3};
  // CHECK: store <4 x i8> <i8 6, i8 1, i8 8, i8 9>
  constexpr auto c = FourCharsVecSize{8, 4, 2, 1} *
                     FourCharsVecSize{3, 4, 5, 6};
  // CHECK: store <4 x i8> <i8 24, i8 16, i8 10, i8 6>
  constexpr auto d = FourCharsVecSize{12, 12, 10, 10} /
                     FourCharsVecSize{6, 4, 5, 2};
  // CHECK: store <4 x i8> <i8 2, i8 3, i8 2, i8 5>
  constexpr auto e = FourCharsVecSize{12, 12, 10, 10} %
                     FourCharsVecSize{6, 4, 4, 3};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 2, i8 1>

  constexpr auto f = FourCharsVecSize{6, 3, 2, 1} + 3;
  // CHECK: store <4 x i8> <i8 9, i8 6, i8 5, i8 4>
  constexpr auto g = FourCharsVecSize{19, 15, 12, 10} - 3;
  // CHECK: store <4 x i8> <i8 16, i8 12, i8 9, i8 7>
  constexpr auto h = FourCharsVecSize{8, 4, 2, 1} * 3;
  // CHECK: store <4 x i8> <i8 24, i8 12, i8 6, i8 3>
  constexpr auto j = FourCharsVecSize{12, 15, 18, 21} / 3;
  // CHECK: store <4 x i8> <i8 4, i8 5, i8 6, i8 7>
  constexpr auto k = FourCharsVecSize{12, 17, 19, 22} % 3;
  // CHECK: store <4 x i8> <i8 0, i8 2, i8 1, i8 1>

  constexpr auto l = 3 + FourCharsVecSize{6, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 9, i8 6, i8 5, i8 4>
  constexpr auto m = 20 - FourCharsVecSize{19, 15, 12, 10};
  // CHECK: store <4 x i8> <i8 1, i8 5, i8 8, i8 10>
  constexpr auto n = 3 * FourCharsVecSize{8, 4, 2, 1};
  // CHECK: store <4 x i8> <i8 24, i8 12, i8 6, i8 3>
  constexpr auto o = 100 / FourCharsVecSize{12, 15, 18, 21};
  // CHECK: store <4 x i8> <i8 8, i8 6, i8 5, i8 4>
  constexpr auto p = 100 % FourCharsVecSize{12, 15, 18, 21};
  // CHECK: store <4 x i8> <i8 4, i8 10, i8 10, i8 16>

  constexpr auto q = FourCharsVecSize{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  // CHECK: store <4 x i8> <i8 12, i8 6, i8 8, i8 4>
  constexpr auto r = FourCharsVecSize{19, 15, 12, 10} >>
                     FourCharsVecSize{1, 1, 2, 2};
  // CHECK: store <4 x i8> <i8 9, i8 7, i8 3, i8 2>
  constexpr auto s = FourCharsVecSize{6, 3, 5, 10} << 1;
  // CHECK: store <4 x i8> <i8 12, i8 6, i8 10, i8 20>
  constexpr auto t = FourCharsVecSize{19, 15, 10, 20} >> 1;
  // CHECK: store <4 x i8> <i8 9, i8 7, i8 5, i8 10>
  constexpr auto u = 12 << FourCharsVecSize{1, 2, 3, 3};
  // CHECK: store <4 x i8> <i8 24, i8 48, i8 96, i8 96>
  constexpr auto v = 12 >> FourCharsVecSize{1, 2, 2, 1};
  // CHECK: store <4 x i8> <i8 6, i8 3, i8 3, i8 6>

  constexpr auto w = FourCharsVecSize{1, 2, 3, 4} <
                     FourCharsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 0>
  constexpr auto x = FourCharsVecSize{1, 2, 3, 4} >
                     FourCharsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto y = FourCharsVecSize{1, 2, 3, 4} <=
                     FourCharsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 0>
  constexpr auto z = FourCharsVecSize{1, 2, 3, 4} >=
                     FourCharsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto A = FourCharsVecSize{1, 2, 3, 4} ==
                     FourCharsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 0>
  constexpr auto B = FourCharsVecSize{1, 2, 3, 4} !=
                     FourCharsVecSize{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 -1>

  constexpr auto C = FourCharsVecSize{1, 2, 3, 4} < 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 0>
  constexpr auto D = FourCharsVecSize{1, 2, 3, 4} > 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 0, i8 -1>
  constexpr auto E = FourCharsVecSize{1, 2, 3, 4} <= 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 0>
  constexpr auto F = FourCharsVecSize{1, 2, 3, 4} >= 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto G = FourCharsVecSize{1, 2, 3, 4} == 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 0>
  constexpr auto H = FourCharsVecSize{1, 2, 3, 4} != 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 -1>

  constexpr auto I = FourCharsVecSize{1, 2, 3, 4} &
                     FourCharsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 0, i8 2, i8 2, i8 0>
  constexpr auto J = FourCharsVecSize{1, 2, 3, 4} ^
                     FourCharsVecSize { 4, 3, 2, 1 };
  // CHECK: store <4 x i8> <i8 5, i8 1, i8 1, i8 5>
  constexpr auto K = FourCharsVecSize{1, 2, 3, 4} |
                     FourCharsVecSize{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 5, i8 3, i8 3, i8 5>
  constexpr auto L = FourCharsVecSize{1, 2, 3, 4} & 3;
  // CHECK: store <4 x i8> <i8 1, i8 2, i8 3, i8 0>
  constexpr auto M = FourCharsVecSize{1, 2, 3, 4} ^ 3;
  // CHECK: store <4 x i8> <i8 2, i8 1, i8 0, i8 7>
  constexpr auto N = FourCharsVecSize{1, 2, 3, 4} | 3;
  // CHECK: store <4 x i8> <i8 3, i8 3, i8 3, i8 7>

  constexpr auto O = FourCharsVecSize{5, 0, 6, 0} &&
                     FourCharsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i8> <i8 1, i8 0, i8 0, i8 0>
  constexpr auto P = FourCharsVecSize{5, 0, 6, 0} ||
                     FourCharsVecSize{5, 5, 0, 0};
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 1, i8 0>

  constexpr auto Q = FourCharsVecSize{5, 0, 6, 0} && 3;
  // CHECK: store <4 x i8> <i8 1, i8 0, i8 1, i8 0>
  constexpr auto R = FourCharsVecSize{5, 0, 6, 0} || 3;
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 1, i8 1>

  constexpr auto T = CmpMul(a, b);
  // CHECK: store <4 x i8> <i8 108, i8 18, i8 56, i8 72>

  constexpr auto U = CmpDiv(a, b);
  // CHECK: store <4 x i8> <i8 3, i8 18, i8 0, i8 0>

  constexpr auto V = CmpRem(a, b);
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 7, i8 8>

  constexpr auto X = CmpAdd(a, b);
  // CHECK: store <4 x i8> <i8 24, i8 19, i8 15, i8 17>

  constexpr auto Y = CmpSub(a, b);
  // CHECK: store <4 x i8> <i8 12, i8 17, i8 -1, i8 -1>

  constexpr auto InvH = -H;
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 0, i8 1>
  constexpr auto Z = CmpLSH(a, InvH);
  // CHECK: store <4 x i8> <i8 36, i8 36, i8 7, i8 16>

  constexpr auto aa = CmpRSH(a, InvH);
  // CHECK: store <4 x i8> <i8 9, i8 9, i8 7, i8 4>

  constexpr auto ab = CmpBinAnd(a, b);
  // CHECK: store <4 x i8> <i8 2, i8 0, i8 0, i8 8>

  constexpr auto ac = CmpBinXOr(a, b);
  // CHECK: store <4 x i8> <i8 20, i8 19, i8 15, i8 1>

  constexpr auto ad = CmpBinOr(a, b);
  // CHECK: store <4 x i8> <i8 22, i8 19, i8 15, i8 9>

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  // CHECK: store <4 x i8> <i8 -2, i8 -3, i8 -11, i8 -21>

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  // CHECK: store <4 x i8> <i8 -1, i8 0, i8 0, i8 0>
}

void CharExtVecUsage() {
  constexpr auto a = FourCharsExtVec{6, 3, 2, 1} +
                     FourCharsExtVec{12, 15, 5, 7};
  // CHECK: store <4 x i8> <i8 18, i8 18, i8 7, i8 8>
  constexpr auto b = FourCharsExtVec{19, 15, 13, 12} -
                     FourCharsExtVec{13, 14, 5, 3};
  // CHECK: store <4 x i8> <i8 6, i8 1, i8 8, i8 9>
  constexpr auto c = FourCharsExtVec{8, 4, 2, 1} *
                     FourCharsExtVec{3, 4, 5, 6};
  // CHECK: store <4 x i8> <i8 24, i8 16, i8 10, i8 6>
  constexpr auto d = FourCharsExtVec{12, 12, 10, 10} /
                     FourCharsExtVec{6, 4, 5, 2};
  // CHECK: store <4 x i8> <i8 2, i8 3, i8 2, i8 5>
  constexpr auto e = FourCharsExtVec{12, 12, 10, 10} %
                     FourCharsExtVec{6, 4, 4, 3};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 2, i8 1>

  constexpr auto f = FourCharsExtVec{6, 3, 2, 1} + 3;
  // CHECK: store <4 x i8> <i8 9, i8 6, i8 5, i8 4>
  constexpr auto g = FourCharsExtVec{19, 15, 12, 10} - 3;
  // CHECK: store <4 x i8> <i8 16, i8 12, i8 9, i8 7>
  constexpr auto h = FourCharsExtVec{8, 4, 2, 1} * 3;
  // CHECK: store <4 x i8> <i8 24, i8 12, i8 6, i8 3>
  constexpr auto j = FourCharsExtVec{12, 15, 18, 21} / 3;
  // CHECK: store <4 x i8> <i8 4, i8 5, i8 6, i8 7>
  constexpr auto k = FourCharsExtVec{12, 17, 19, 22} % 3;
  // CHECK: store <4 x i8> <i8 0, i8 2, i8 1, i8 1>

  constexpr auto l = 3 + FourCharsExtVec{6, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 9, i8 6, i8 5, i8 4>
  constexpr auto m = 20 - FourCharsExtVec{19, 15, 12, 10};
  // CHECK: store <4 x i8> <i8 1, i8 5, i8 8, i8 10>
  constexpr auto n = 3 * FourCharsExtVec{8, 4, 2, 1};
  // CHECK: store <4 x i8> <i8 24, i8 12, i8 6, i8 3>
  constexpr auto o = 100 / FourCharsExtVec{12, 15, 18, 21};
  // CHECK: store <4 x i8> <i8 8, i8 6, i8 5, i8 4>
  constexpr auto p = 100 % FourCharsExtVec{12, 15, 18, 21};
  // CHECK: store <4 x i8> <i8 4, i8 10, i8 10, i8 16>

  constexpr auto q = FourCharsExtVec{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  // CHECK: store <4 x i8> <i8 12, i8 6, i8 8, i8 4>
  constexpr auto r = FourCharsExtVec{19, 15, 12, 10} >>
                     FourCharsExtVec{1, 1, 2, 2};
  // CHECK: store <4 x i8> <i8 9, i8 7, i8 3, i8 2>
  constexpr auto s = FourCharsExtVec{6, 3, 5, 10} << 1;
  // CHECK: store <4 x i8> <i8 12, i8 6, i8 10, i8 20>
  constexpr auto t = FourCharsExtVec{19, 15, 10, 20} >> 1;
  // CHECK: store <4 x i8> <i8 9, i8 7, i8 5, i8 10>
  constexpr auto u = 12 << FourCharsExtVec{1, 2, 3, 3};
  // CHECK: store <4 x i8> <i8 24, i8 48, i8 96, i8 96>
  constexpr auto v = 12 >> FourCharsExtVec{1, 2, 2, 1};
  // CHECK: store <4 x i8> <i8 6, i8 3, i8 3, i8 6>

  constexpr auto w = FourCharsExtVec{1, 2, 3, 4} <
                     FourCharsExtVec{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 0>
  constexpr auto x = FourCharsExtVec{1, 2, 3, 4} >
                     FourCharsExtVec{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto y = FourCharsExtVec{1, 2, 3, 4} <=
                     FourCharsExtVec{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 0>
  constexpr auto z = FourCharsExtVec{1, 2, 3, 4} >=
                     FourCharsExtVec{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto A = FourCharsExtVec{1, 2, 3, 4} ==
                     FourCharsExtVec{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 0>
  constexpr auto B = FourCharsExtVec{1, 2, 3, 4} !=
                     FourCharsExtVec{4, 3, 3, 1};
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 -1>

  constexpr auto C = FourCharsExtVec{1, 2, 3, 4} < 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 0>
  constexpr auto D = FourCharsExtVec{1, 2, 3, 4} > 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 0, i8 -1>
  constexpr auto E = FourCharsExtVec{1, 2, 3, 4} <= 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 -1, i8 0>
  constexpr auto F = FourCharsExtVec{1, 2, 3, 4} >= 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 -1>
  constexpr auto G = FourCharsExtVec{1, 2, 3, 4} == 3;
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 -1, i8 0>
  constexpr auto H = FourCharsExtVec{1, 2, 3, 4} != 3;
  // CHECK: store <4 x i8> <i8 -1, i8 -1, i8 0, i8 -1>

  constexpr auto I = FourCharsExtVec{1, 2, 3, 4} &
                     FourCharsExtVec{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 0, i8 2, i8 2, i8 0>
  constexpr auto J = FourCharsExtVec{1, 2, 3, 4} ^
                     FourCharsExtVec { 4, 3, 2, 1 };
  // CHECK: store <4 x i8> <i8 5, i8 1, i8 1, i8 5>
  constexpr auto K = FourCharsExtVec{1, 2, 3, 4} |
                     FourCharsExtVec{4, 3, 2, 1};
  // CHECK: store <4 x i8> <i8 5, i8 3, i8 3, i8 5>
  constexpr auto L = FourCharsExtVec{1, 2, 3, 4} & 3;
  // CHECK: store <4 x i8> <i8 1, i8 2, i8 3, i8 0>
  constexpr auto M = FourCharsExtVec{1, 2, 3, 4} ^ 3;
  // CHECK: store <4 x i8> <i8 2, i8 1, i8 0, i8 7>
  constexpr auto N = FourCharsExtVec{1, 2, 3, 4} | 3;
  // CHECK: store <4 x i8> <i8 3, i8 3, i8 3, i8 7>

  constexpr auto O = FourCharsExtVec{5, 0, 6, 0} &&
                     FourCharsExtVec{5, 5, 0, 0};
  // CHECK: store <4 x i8> <i8 1, i8 0, i8 0, i8 0>
  constexpr auto P = FourCharsExtVec{5, 0, 6, 0} ||
                     FourCharsExtVec{5, 5, 0, 0};
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 1, i8 0>

  constexpr auto Q = FourCharsExtVec{5, 0, 6, 0} && 3;
  // CHECK: store <4 x i8> <i8 1, i8 0, i8 1, i8 0>
  constexpr auto R = FourCharsExtVec{5, 0, 6, 0} || 3;
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 1, i8 1>

  constexpr auto T = CmpMul(a, b);
  // CHECK: store <4 x i8> <i8 108, i8 18, i8 56, i8 72>

  constexpr auto U = CmpDiv(a, b);
  // CHECK: store <4 x i8> <i8 3, i8 18, i8 0, i8 0>

  constexpr auto V = CmpRem(a, b);
  // CHECK: store <4 x i8> <i8 0, i8 0, i8 7, i8 8>

  constexpr auto X = CmpAdd(a, b);
  // CHECK: store <4 x i8> <i8 24, i8 19, i8 15, i8 17>

  constexpr auto Y = CmpSub(a, b);
  // CHECK: store <4 x i8> <i8 12, i8 17, i8 -1, i8 -1>

  constexpr auto InvH = -H;
  // CHECK: store <4 x i8> <i8 1, i8 1, i8 0, i8 1>

  constexpr auto Z = CmpLSH(a, InvH);
  // CHECK: store <4 x i8> <i8 36, i8 36, i8 7, i8 16>

  constexpr auto aa = CmpRSH(a, InvH);
  // CHECK: store <4 x i8> <i8 9, i8 9, i8 7, i8 4>

  constexpr auto ab = CmpBinAnd(a, b);
  // CHECK: store <4 x i8> <i8 2, i8 0, i8 0, i8 8>

  constexpr auto ac = CmpBinXOr(a, b);
  // CHECK: store <4 x i8> <i8 20, i8 19, i8 15, i8 1>

  constexpr auto ad = CmpBinOr(a, b);
  // CHECK: store <4 x i8> <i8 22, i8 19, i8 15, i8 9>

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  // CHECK: store <4 x i8> <i8 -2, i8 -3, i8 -11, i8 -21>

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  // CHECK: store <4 x i8> <i8 -1, i8 0, i8 0, i8 0>
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
  // CHECK: store <4 x i128> <i128 1, i128 2, i128 3, i128 4>
  constexpr auto b = a < 3;
  // CHECK: store <4 x i128> <i128 -1, i128 -1, i128 0, i128 0>

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128VecSize{1, 2, 10, 20};
  // CHECK: store <4 x i128> <i128 -2, i128 -3, i128 -11, i128 -21>

  constexpr auto d = !FourI128VecSize{0, 1, 8, -1};
  // CHECK: store <4 x i128> <i128 -1, i128 0, i128 0, i128 0>
}

void I128VecUsage() {
  constexpr auto a = FourI128ExtVec{1, 2, 3, 4};
  // CHECK: store <4 x i128> <i128 1, i128 2, i128 3, i128 4>
  constexpr auto b = a < 3;
  // CHECK: store <4 x i128> <i128 -1, i128 -1, i128 0, i128 0>

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128ExtVec{1, 2, 10, 20};
  // CHECK: store <4 x i128>  <i128 -2, i128 -3, i128 -11, i128 -21>

  constexpr auto d = !FourI128ExtVec{0, 1, 8, -1};
  // CHECK: store <4 x i128>  <i128 -1, i128 0, i128 0, i128 0>
}

using FourBoolsExtVec __attribute__((ext_vector_type(4))) = bool;
void BoolVecUsage() {
  constexpr auto a = FourBoolsExtVec{true, false, true, false} <
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %a, align 1
  constexpr auto b = FourBoolsExtVec{true, false, true, false} <=
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 true, i1 true, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %b, align 1
  constexpr auto c = FourBoolsExtVec{true, false, true, false} ==
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 true, i1 true, i1 false, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %c, align 1
  constexpr auto d = FourBoolsExtVec{true, false, true, false} !=
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 true, i1 false, i1 false, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %d, align 1
  constexpr auto e = FourBoolsExtVec{true, false, true, false} >=
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %e, align 1
  constexpr auto f = FourBoolsExtVec{true, false, true, false} >
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %f, align 1
  constexpr auto g = FourBoolsExtVec{true, false, true, false} &
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 false, i1 true, i1 false, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %g, align 1
  constexpr auto h = FourBoolsExtVec{true, false, true, false} |
                     FourBoolsExtVec{false, false, true, true};
  // CHECK: store i8 bitcast (<8 x i1> <i1 true, i1 false, i1 true, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %h, align 1
  constexpr auto i = FourBoolsExtVec{true, false, true, false} ^
                     FourBoolsExtVec { false, false, true, true };
  // CHECK: store i8 bitcast (<8 x i1> <i1 true, i1 false, i1 false, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %i, align 1
  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 true, i1 false, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %j, align 1
  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  // CHECK: store i8 bitcast (<8 x i1> <i1 false, i1 true, i1 false, i1 true, i1 undef, i1 undef, i1 undef, i1 undef> to i8), i8* %k, align 1
}
