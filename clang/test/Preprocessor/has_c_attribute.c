// RUN: %clang_cc1 -std=c11 -E -P %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c2x -E -P %s -o - | FileCheck %s

#define C2x(x) x: __has_c_attribute(x)

// CHECK: fallthrough: 201910L
C2x(fallthrough)

// CHECK: __nodiscard__: 202003L
C2x(__nodiscard__)

// CHECK: selectany: 0
C2x(selectany); // Known attribute not supported in C mode

// CHECK: frobble: 0
C2x(frobble) // Unknown attribute

// CHECK: frobble::frobble: 0
C2x(frobble::frobble) // Unknown vendor namespace

// CHECK: clang::annotate: 1
C2x(clang::annotate)

// CHECK: deprecated: 201904L
C2x(deprecated)

// CHECK: maybe_unused: 202106L
C2x(maybe_unused)

// CHECK: __gnu__::warn_unused_result: 202003L
C2x(__gnu__::warn_unused_result)

// CHECK: gnu::__warn_unused_result__: 202003L
C2x(gnu::__warn_unused_result__)

// Test that macro expansion of the builtin argument works.
#define C clang
#define L likely
#define CL clang::likely
#define N nodiscard

#if __has_c_attribute(N)
int has_nodiscard;
#endif
// CHECK: int has_nodiscard;

#if __has_c_attribute(C::L)
int has_clang_likely_1;
#endif
// CHECK: int has_clang_likely_1;

#if __has_c_attribute(clang::L)
int has_clang_likely_2;
#endif
// CHECK: int has_clang_likely_2;

#if __has_c_attribute(C::likely)
int has_clang_likely_3;
#endif
// CHECK: int has_clang_likely_3;

#if __has_c_attribute(CL)
int has_clang_likely_4;
#endif
// CHECK: int has_clang_likely_4;

#define FUNCLIKE1(x) clang::x
#if __has_c_attribute(FUNCLIKE1(likely))
int funclike_1;
#endif
// CHECK: int funclike_1;

#define FUNCLIKE2(x) _Clang::x
#if __has_c_attribute(FUNCLIKE2(likely))
int funclike_2;
#endif
// CHECK: int funclike_2;
