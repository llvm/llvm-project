// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z15 -fzvector -flax-vector-conversions=none -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z15 -fzvector -flax-vector-conversions=none -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z15 -fzvector -flax-vector-conversions=none -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z15 -fzvector -flax-vector-conversions=none -std=c++17 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

// Wrappers whose body has more than one statement are only constexpr from
// C++14 on; their asserts are guarded with __cplusplus >= 201402L.

#include <vecintrin.h>

//===----------------------------------------------------------------------===//
// Helpers (C++11-compatible: single return statement, recursion)
//===----------------------------------------------------------------------===//

// True if every lane of v equals x (x is converted to the lane type).
template <typename V, typename T>
constexpr bool all_lanes(V v, T x, unsigned i = 0) {
  return i == sizeof(V) / sizeof(v[0])
             ? true
             : (v[i] == (__typeof__(v[0]))x && all_lanes(v, x, i + 1));
}

// True if lane i, i+1, ... of v equal the listed values in order.
template <typename V>
constexpr bool lanes_from(V, unsigned) { return true; }
template <typename V, typename T, typename... Ts>
constexpr bool lanes_from(V v, unsigned i, T x, Ts... xs) {
  return v[i] == (__typeof__(v[0]))x && lanes_from(v, i + 1, xs...);
}
#define LANES(v, ...) lanes_from((v), 0u, __VA_ARGS__)

#define V2(x)  {x, x}
#define V4(x)  {x, x, x, x}
#define V8(x)  {x, x, x, x, x, x, x, x}
#define V16(x) {x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}

//===----------------------------------------------------------------------===//
// Test data
//===----------------------------------------------------------------------===//

// Distinct lanes, for the position-dependent intrinsics.
constexpr __vector signed char  ca = {1, 2, 3, 4, 5, 6, 7, 8,
                                      9, 10, 11, 12, 13, 14, 15, 16};
constexpr __vector signed char  cb = {21, 22, 23, 24, 25, 26, 27, 28,
                                      29, 30, 31, 32, 33, 34, 35, 36};
constexpr __vector signed short sa = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr __vector signed short sb = {11, 12, 13, 14, 15, 16, 17, 18};
constexpr __vector signed int   ia = {1, 2, 3, 4};
constexpr __vector signed int   ib = {10, 20, 30, 40};
constexpr __vector signed int   ic = {1, 0, 3, 9};
constexpr __vector unsigned int uia = {1, 2, 3, 4};
constexpr __vector unsigned int uib = {10, 20, 30, 40};
constexpr __vector signed long long la = {1, 2};
constexpr __vector signed long long lb = {10, 20};
constexpr __vector float        fa = {1.5f, 2.5f, 3.5f, 4.5f};
constexpr __vector float        fb = {10.5f, 20.5f, 30.5f, 40.5f};
constexpr __vector double       da = {1.5, 2.5};
constexpr __vector double       db = {10.5, 20.5};

// Same value in every lane, for the lane-wise intrinsics.
// 12 = 0b1100, 10 = 0b1010.
constexpr __vector signed char  c12 = V16(12);
constexpr __vector signed char  c10 = V16(10);
constexpr __vector signed short s12 = V8(12);
constexpr __vector signed short s10 = V8(10);
constexpr __vector signed int   i12 = V4(12);
constexpr __vector signed int   i10 = V4(10);
constexpr __vector signed long long l12 = V2(12);
constexpr __vector signed long long l10 = V2(10);
constexpr __vector unsigned char  uc12 = V16(12);
constexpr __vector unsigned char  uc10 = V16(10);
constexpr __vector unsigned int   ui12 = V4(12u);
constexpr __vector unsigned int   ui10 = V4(10u);

// Boolean vectors, built with vec_cmpeq / vec_cmplt on equal operands.
constexpr __vector __bool char  bct = vec_cmpeq(c12, c12);  // all true
constexpr __vector __bool int   bit = vec_cmpeq(i12, i12);  // all true
constexpr __vector __bool int   bif = vec_cmplt(i12, i12);  // all false

//===----------------------------------------------------------------------===//
// vec_splats
//===----------------------------------------------------------------------===//
static_assert(all_lanes(vec_splats((signed char)-7), -7), "");
static_assert(all_lanes(vec_splats((unsigned char)200), 200), "");
static_assert(all_lanes(vec_splats((signed short)-300), -300), "");
static_assert(all_lanes(vec_splats((unsigned short)60000), 60000), "");
static_assert(all_lanes(vec_splats(-100000), -100000), "");
static_assert(all_lanes(vec_splats(4000000000u), 4000000000u), "");
static_assert(all_lanes(vec_splats(-5000000000LL), -5000000000LL), "");
static_assert(all_lanes(vec_splats(10000000000ULL), 10000000000ULL), "");
static_assert(all_lanes(vec_splats(1.5f), 1.5f), "");
static_assert(all_lanes(vec_splats(-2.25), -2.25), "");
static_assert(all_lanes(vec_splats(0), 0), "");

//===----------------------------------------------------------------------===//
// vec_splat_s8/16/32/64 and vec_splat_u8/16/32/64
//===----------------------------------------------------------------------===//
static_assert(all_lanes(vec_splat_s8(-3), -3), "");
static_assert(all_lanes(vec_splat_s16(-3), -3), "");
static_assert(all_lanes(vec_splat_s32(-3), -3), "");
static_assert(all_lanes(vec_splat_s64(-3), -3), "");
static_assert(all_lanes(vec_splat_u8(200), 200), "");
static_assert(all_lanes(vec_splat_u16(65535), 65535), "");
static_assert(all_lanes(vec_splat_u32(-1), 0xFFFFFFFFu), "");
static_assert(all_lanes(vec_splat_u64(-1), 0xFFFFFFFFFFFFFFFFULL), "");

//===----------------------------------------------------------------------===//
// vec_and / vec_or / vec_xor / vec_andc / vec_nor
//===----------------------------------------------------------------------===//
static_assert(all_lanes(vec_and(c12, c10), 8), "");
static_assert(all_lanes(vec_and(s12, s10), 8), "");
static_assert(all_lanes(vec_and(i12, i10), 8), "");
static_assert(all_lanes(vec_and(l12, l10), 8), "");
static_assert(all_lanes(vec_and(uc12, uc10), 8), "");
static_assert(all_lanes(vec_and(ui12, ui10), 8), "");

static_assert(all_lanes(vec_or(c12, c10), 14), "");
static_assert(all_lanes(vec_or(s12, s10), 14), "");
static_assert(all_lanes(vec_or(i12, i10), 14), "");
static_assert(all_lanes(vec_or(l12, l10), 14), "");
static_assert(all_lanes(vec_or(ui12, ui10), 14), "");

static_assert(all_lanes(vec_xor(c12, c10), 6), "");
static_assert(all_lanes(vec_xor(s12, s10), 6), "");
static_assert(all_lanes(vec_xor(i12, i10), 6), "");
static_assert(all_lanes(vec_xor(l12, l10), 6), "");
static_assert(all_lanes(vec_xor(ui12, ui10), 6), "");

static_assert(all_lanes(vec_andc(c12, c10), 4), "");    // 12 & ~10
static_assert(all_lanes(vec_andc(s12, s10), 4), "");
static_assert(all_lanes(vec_andc(i12, i10), 4), "");
static_assert(all_lanes(vec_andc(l12, l10), 4), "");
static_assert(all_lanes(vec_andc(ui12, ui10), 4), "");

static_assert(all_lanes(vec_nor(c12, c10), -15), "");   // ~(12 | 10)
static_assert(all_lanes(vec_nor(s12, s10), -15), "");
static_assert(all_lanes(vec_nor(i12, i10), -15), "");
static_assert(all_lanes(vec_nor(l12, l10), -15), "");
static_assert(all_lanes(vec_nor(uc12, uc10), 0xF1), "");
static_assert(all_lanes(vec_nor(ui12, ui10), 0xFFFFFFF1u), "");

// Boolean overloads.
static_assert(all_lanes(vec_and(bit, bif), 0), "");
static_assert(all_lanes(vec_or(bit, bif), -1), "");
static_assert(all_lanes(vec_xor(bit, bit), 0), "");
static_assert(all_lanes(vec_andc(bit, bit), 0), "");
static_assert(all_lanes(vec_nor(bif, bif), -1), "");
static_assert(all_lanes(vec_and(bct, bct), -1), "");

// Floating-point overloads work on the bit pattern.
static_assert(all_lanes(vec_xor(fa, fa), 0.0f), "");
static_assert(LANES(vec_and(fa, fa), 1.5f, 2.5f, 3.5f, 4.5f), "");
static_assert(LANES(vec_or(da, da), 1.5, 2.5), "");
static_assert(all_lanes(vec_xor(da, da), 0.0), "");

//===----------------------------------------------------------------------===//
// vec_cmpeq / vec_cmpgt / vec_cmpge / vec_cmplt / vec_cmple
// Results are boolean vectors: all-ones (true) or zero (false) per lane.
//===----------------------------------------------------------------------===//
static_assert(all_lanes(vec_cmpeq(c12, c10), 0), "");
static_assert(all_lanes(vec_cmpeq(c12, c12), -1), "");
static_assert(all_lanes(vec_cmpeq(s12, s12), -1), "");
static_assert(all_lanes(vec_cmpeq(i12, i10), 0), "");
static_assert(all_lanes(vec_cmpeq(l12, l12), -1), "");

static_assert(all_lanes(vec_cmpgt(c12, c10), -1), "");
static_assert(all_lanes(vec_cmpgt(s12, s10), -1), "");
static_assert(all_lanes(vec_cmpgt(i10, i12), 0), "");
static_assert(all_lanes(vec_cmpgt(l12, l10), -1), "");
static_assert(all_lanes(vec_cmpgt(uc12, uc10), -1), "");
static_assert(all_lanes(vec_cmpgt(ui10, ui12), 0), "");

static_assert(all_lanes(vec_cmpge(c12, c10), -1), "");
static_assert(all_lanes(vec_cmpge(i12, i12), -1), "");
static_assert(all_lanes(vec_cmpge(i10, i12), 0), "");
static_assert(all_lanes(vec_cmpge(l12, l10), -1), "");

static_assert(all_lanes(vec_cmplt(c10, c12), -1), "");
static_assert(all_lanes(vec_cmplt(s12, s10), 0), "");
static_assert(all_lanes(vec_cmplt(i10, i12), -1), "");
static_assert(all_lanes(vec_cmplt(l12, l12), 0), "");

static_assert(all_lanes(vec_cmple(c10, c12), -1), "");
static_assert(all_lanes(vec_cmple(c12, c12), -1), "");
static_assert(all_lanes(vec_cmple(s12, s10), 0), "");
static_assert(all_lanes(vec_cmple(i10, i12), -1), "");

// Lanes that differ.
static_assert(LANES(vec_cmpeq(ia, ic), -1, 0, -1, 0), "");
static_assert(LANES(vec_cmpgt(ia, ic), 0, -1, 0, 0), "");
static_assert(LANES(vec_cmplt(ia, ic), 0, 0, 0, -1), "");

// Floating point.
static_assert(all_lanes(vec_cmpeq(fa, fa), -1), "");
static_assert(all_lanes(vec_cmplt(fa, fb), -1), "");
static_assert(all_lanes(vec_cmpgt(fa, fb), 0), "");
static_assert(all_lanes(vec_cmple(da, db), -1), "");
static_assert(all_lanes(vec_cmpge(da, db), 0), "");

//===----------------------------------------------------------------------===//
// vec_sel: take bits from b where the mask is 1, from a where it is 0.
//===----------------------------------------------------------------------===//
constexpr __vector unsigned int ua_bits = V4(0x0000FFFFu);
constexpr __vector unsigned int ub_bits = V4(0xFFFF0000u);
constexpr __vector unsigned int umask   = V4(0xFF00FF00u);
constexpr __vector unsigned int ulane_m = {0xFFFFFFFFu, 0, 0xFFFFFFFFu, 0};
constexpr __vector unsigned long long ulm = {~0ULL, 0};

static_assert(all_lanes(vec_sel(ua_bits, ub_bits, umask), 0xFF0000FFu), "");
static_assert(all_lanes(vec_sel(c12, c10, bct), 10), "");
static_assert(all_lanes(vec_sel(c12, c10, vec_cmplt(c12, c12)), 12), "");
static_assert(LANES(vec_sel(ia, ib, vec_cmpeq(ia, ic)), 10, 2, 30, 4), "");
static_assert(LANES(vec_sel(ia, ib, ulane_m), 10, 2, 30, 4), "");
static_assert(all_lanes(vec_sel(s12, s10, vec_cmpeq(s12, s12)), 10), "");
static_assert(all_lanes(vec_sel(l12, l10, vec_cmpeq(l12, l12)), 10), "");
static_assert(LANES(vec_sel(da, db, ulm), 10.5, 2.5), "");
static_assert(LANES(vec_sel(fa, fb, ulane_m), 10.5f, 2.5f, 30.5f, 4.5f), "");
static_assert(all_lanes(vec_sel(bif, bit, bit), -1), "");

//===----------------------------------------------------------------------===//
// vec_abs (integer forms; the float forms call builtins and are Tier 2)
//===----------------------------------------------------------------------===//
constexpr __vector signed int iabs = {-5, 7, 0, -1};
static_assert(LANES(vec_abs(iabs), 5, 7, 0, 1), "");
static_assert(all_lanes(vec_abs((__vector signed char)V16(-5)), 5), "");
static_assert(all_lanes(vec_abs((__vector signed short)V8(-5)), 5), "");
static_assert(all_lanes(vec_abs((__vector signed long long)V2(-5)), 5), "");

//===----------------------------------------------------------------------===//
// vec_max / vec_min (integer forms; the float forms call builtins)
//===----------------------------------------------------------------------===//
constexpr __vector unsigned int ux = {0xFFFFFFFFu, 1, 5, 0};
constexpr __vector unsigned int uy = {1, 0xFFFFFFFFu, 5, 0};

static_assert(all_lanes(vec_max(c12, c10), 12), "");
static_assert(all_lanes(vec_min(c12, c10), 10), "");
static_assert(all_lanes(vec_max(s10, s12), 12), "");
static_assert(all_lanes(vec_min(s10, s12), 10), "");
static_assert(all_lanes(vec_max(l12, l10), 12), "");
static_assert(all_lanes(vec_min(l12, l10), 10), "");
static_assert(LANES(vec_max(ia, ic), 1, 2, 3, 9), "");
static_assert(LANES(vec_min(ia, ic), 1, 0, 3, 4), "");
static_assert(all_lanes(vec_max(uc12, uc10), 12), "");
static_assert(all_lanes(vec_min(uc12, uc10), 10), "");
// Unsigned comparison: 0xFFFFFFFF must win as the maximum.
static_assert(LANES(vec_max(ux, uy), 0xFFFFFFFFu, 0xFFFFFFFFu, 5u, 0u), "");
static_assert(LANES(vec_min(ux, uy), 1u, 1u, 5u, 0u), "");

//===----------------------------------------------------------------------===//
// vec_mergeh / vec_mergel
//===----------------------------------------------------------------------===//
static_assert(LANES(vec_mergeh(ia, ib), 1, 10, 2, 20), "");
static_assert(LANES(vec_mergel(ia, ib), 3, 30, 4, 40), "");
static_assert(LANES(vec_mergeh(uia, uib), 1, 10, 2, 20), "");
static_assert(LANES(vec_mergel(uia, uib), 3, 30, 4, 40), "");
static_assert(LANES(vec_mergeh(sa, sb), 1, 11, 2, 12, 3, 13, 4, 14), "");
static_assert(LANES(vec_mergel(sa, sb), 5, 15, 6, 16, 7, 17, 8, 18), "");
static_assert(LANES(vec_mergeh(la, lb), 1, 10), "");
static_assert(LANES(vec_mergel(la, lb), 2, 20), "");
static_assert(LANES(vec_mergeh(ca, cb), 1, 21, 2, 22, 3, 23, 4, 24,
                    5, 25, 6, 26, 7, 27, 8, 28), "");
static_assert(LANES(vec_mergel(ca, cb), 9, 29, 10, 30, 11, 31, 12, 32,
                    13, 33, 14, 34, 15, 35, 16, 36), "");
static_assert(LANES(vec_mergeh(fa, fb), 1.5f, 10.5f, 2.5f, 20.5f), "");
static_assert(LANES(vec_mergel(fa, fb), 3.5f, 30.5f, 4.5f, 40.5f), "");
static_assert(LANES(vec_mergeh(da, db), 1.5, 10.5), "");
static_assert(LANES(vec_mergel(da, db), 2.5, 20.5), "");
static_assert(LANES(vec_mergeh(bit, bif), -1, 0, -1, 0), "");

//===----------------------------------------------------------------------===//
// vec_splat (lane index is a constant)
//===----------------------------------------------------------------------===//
static_assert(all_lanes(vec_splat(ia, 0), 1), "");
static_assert(all_lanes(vec_splat(ia, 2), 3), "");
static_assert(all_lanes(vec_splat(uia, 3), 4), "");
static_assert(all_lanes(vec_splat(ca, 15), 16), "");
static_assert(all_lanes(vec_splat(sa, 7), 8), "");
static_assert(all_lanes(vec_splat(la, 1), 2), "");
static_assert(all_lanes(vec_splat(fa, 1), 2.5f), "");
static_assert(all_lanes(vec_splat(da, 1), 2.5), "");
static_assert(all_lanes(vec_splat(vec_cmpeq(ia, ic), 0), -1), "");
static_assert(all_lanes(vec_splat(vec_cmpeq(ia, ic), 1), 0), "");

//===----------------------------------------------------------------------===//
// vec_extract (the index is masked to the lane count)
//===----------------------------------------------------------------------===//
static_assert(vec_extract(ia, 0) == 1, "");
static_assert(vec_extract(ia, 3) == 4, "");
static_assert(vec_extract(ia, 4) == 1, "");     // 4 & 3 == 0
static_assert(vec_extract(ia, 7) == 4, "");     // 7 & 3 == 3
static_assert(vec_extract(uia, 2) == 3u, "");
static_assert(vec_extract(ca, 15) == 16, "");
static_assert(vec_extract(ca, 17) == 2, "");    // 17 & 15 == 1
static_assert(vec_extract(sa, 7) == 8, "");
static_assert(vec_extract(la, 1) == 2, "");
static_assert(vec_extract(fa, 2) == 3.5f, "");
static_assert(vec_extract(da, 1) == 2.5, "");
static_assert(vec_extract(vec_cmpeq(ia, ic), 0) == 0xFFFFFFFFu, "");
static_assert(vec_extract(vec_cmpeq(ia, ic), 1) == 0u, "");

//===----------------------------------------------------------------------===//
// vec_reve
//===----------------------------------------------------------------------===//
static_assert(LANES(vec_reve(ia), 4, 3, 2, 1), "");
static_assert(LANES(vec_reve(uia), 4, 3, 2, 1), "");
static_assert(LANES(vec_reve(sa), 8, 7, 6, 5, 4, 3, 2, 1), "");
static_assert(LANES(vec_reve(ca), 16, 15, 14, 13, 12, 11, 10, 9,
                    8, 7, 6, 5, 4, 3, 2, 1), "");
static_assert(LANES(vec_reve(la), 2, 1), "");
static_assert(LANES(vec_reve(fa), 4.5f, 3.5f, 2.5f, 1.5f), "");
static_assert(LANES(vec_reve(da), 2.5, 1.5), "");
static_assert(LANES(vec_reve(vec_cmpeq(ia, ic)), 0, -1, 0, -1), "");

//===----------------------------------------------------------------------===//
// vec_float / vec_double / vec_signed (__builtin_convertvector)
//===----------------------------------------------------------------------===//
constexpr __vector signed long long lcv = {-3, 4};
constexpr __vector unsigned long long ulcv = {1, 2};
constexpr __vector double dcv = {-1.9, 2.9};
constexpr __vector float fcv = {1.5f, -2.5f, 3.9f, -4.9f};

static_assert(LANES(vec_double(lcv), -3.0, 4.0), "");
static_assert(LANES(vec_double(ulcv), 1.0, 2.0), "");
static_assert(LANES(vec_float(ia), 1.0f, 2.0f, 3.0f, 4.0f), "");
static_assert(LANES(vec_float(uia), 1.0f, 2.0f, 3.0f, 4.0f), "");
static_assert(LANES(vec_signed(dcv), -1, 2), "");          // truncates
static_assert(LANES(vec_signed(fcv), 1, -2, 3, -4), "");   // truncates

//===----------------------------------------------------------------------===//
// 128-bit element vectors. These are the least certain asserts: if the
// constant evaluator rejects 128-bit vector elements, move them to a separate
// file rather than weakening the test above.
//===----------------------------------------------------------------------===//
constexpr __vector signed __int128 xa = {(signed __int128)12};
constexpr __vector signed __int128 xb = {(signed __int128)10};
static_assert(vec_and(xa, xb)[0] == 8, "");
static_assert(vec_or(xa, xb)[0] == 14, "");
static_assert(vec_xor(xa, xb)[0] == 6, "");
static_assert(vec_splats((signed __int128)5)[0] == 5, "");

//===----------------------------------------------------------------------===//
// C++14 and later only: the wrappers whose body has more than one statement.
//===----------------------------------------------------------------------===//
#if __cplusplus >= 201402L

// vec_insert: replaces one lane; the index is masked.
static_assert(LANES(vec_insert(99, ia, 2), 1, 2, 99, 4), "");
static_assert(LANES(vec_insert(99, ia, 6), 1, 2, 99, 4), "");   // 6 & 3 == 2
static_assert(LANES(vec_insert((signed char)99, ca, 15),
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99), "");
static_assert(LANES(vec_insert((signed short)99, sa, 0),
                    99, 2, 3, 4, 5, 6, 7, 8), "");
static_assert(LANES(vec_insert(99LL, la, 1), 1, 99), "");
static_assert(LANES(vec_insert(9.5f, fa, 3), 1.5f, 2.5f, 3.5f, 9.5f), "");
static_assert(LANES(vec_insert(9.5, da, 1), 1.5, 9.5), "");

// Deprecated overloads that take a boolean vector and convert it first.
static_assert(all_lanes(vec_max(c12, bct), 12), "");    // bct converts to -1
static_assert(all_lanes(vec_min(c12, bct), -1), "");

#endif // C++14

//===----------------------------------------------------------------------===//
// The wrappers must still work with run-time arguments.
//===----------------------------------------------------------------------===//
__vector signed int runtime_and(__vector signed int x) {
  return vec_and(x, vec_splats(7));
}