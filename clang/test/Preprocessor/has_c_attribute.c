// RUN: %clang_cc1 -fdouble-square-bracket-attributes -std=c11 -E -P %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c2x -E -P %s -o - | FileCheck %s

#define C2x(x) x: __has_c_attribute(x)

// CHECK: fallthrough: 201904L
C2x(fallthrough)

// CHECK: __nodiscard__: 201904L
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

// CHECK: maybe_unused: 201904L
C2x(maybe_unused)

// CHECK: __gnu__::warn_unused_result: 201904L
C2x(__gnu__::warn_unused_result)

// CHECK: gnu::__warn_unused_result__: 201904L
C2x(gnu::__warn_unused_result__)

// We do somewhat support the __clang__ vendor namespace, but it is a
// predefined macro and thus we encourage users to use _Clang instead.
// Because of this, we do not support __has_c_attribute for that
// vendor namespace.
//
// Note, we can't use C2x here because it will expand __clang__ to 1
// too early.
// CHECK: 1::fallthrough: 0
__clang__::fallthrough: __has_c_attribute(__clang__::fallthrough)
