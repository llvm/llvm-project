// RUN: %clang_cc1 -triple i386-unknown-unknown -fms-compatibility -std=c++11 -E -P %s -o - | FileCheck %s --check-prefixes=CHECK,ITANIUM --implicit-check-not=:
// RUN: %clang_cc1 -triple i386-windows -fms-compatibility -std=c++11 -E -P %s -o - | FileCheck %s --check-prefixes=CHECK,WINDOWS --implicit-check-not=:

#define CXX11(x) x: __has_cpp_attribute(x)

// CHECK: clang::fallthrough: 201603L
CXX11(clang::fallthrough)

// CHECK: selectany: 0
CXX11(selectany)

// The attribute name can be bracketed with double underscores.
// CHECK: clang::__fallthrough__: 201603L
CXX11(clang::__fallthrough__)

// The scope cannot be bracketed with double underscores unless it is
// for gnu or clang.
// CHECK: __gsl__::suppress: 0
CXX11(__gsl__::suppress)

// We do somewhat support the __clang__ vendor namespace, but it is a
// predefined macro and thus we encourage users to use _Clang instead.
// Because of this, we do not support __has_cpp_attribute for that
// vendor namespace.
//
// Note, we can't use CXX11 here because it will expand __clang__ to 1
// too early.
// CHECK: 1::fallthrough: 0
__clang__::fallthrough: __has_cpp_attribute(__clang__::fallthrough)

// CHECK: _Clang::fallthrough: 201603L
CXX11(_Clang::fallthrough)

// CHECK: __nodiscard__: 201907L
CXX11(__nodiscard__)

// CHECK: __gnu__::__const__: 1
CXX11(__gnu__::__const__)

// Test that C++11, target-specific attributes behave properly.

// CHECK: gnu::mips16: 0
CXX11(gnu::mips16)

// Test for standard attributes as listed in C++2a [cpp.cond] paragraph 6.

CXX11(assert)
CXX11(carries_dependency)
CXX11(deprecated)
CXX11(ensures)
CXX11(expects)
CXX11(fallthrough)
CXX11(likely)
CXX11(maybe_unused)
CXX11(no_unique_address)
CXX11(nodiscard)
CXX11(noreturn)
CXX11(unlikely)
// FIXME(201806L) CHECK: assert: 0
// CHECK: carries_dependency: 200809L
// CHECK: deprecated: 201309L
// FIXME(201806L) CHECK: ensures: 0
// FIXME(201806L) CHECK: expects: 0
// CHECK: fallthrough: 201603L
// FIXME(201803L) CHECK: likely: 2L
// CHECK: maybe_unused: 201603L
// ITANIUM: no_unique_address: 201803L
// WINDOWS: no_unique_address: 0
// CHECK: nodiscard: 201907L
// CHECK: noreturn: 200809L
// FIXME(201803L) CHECK: unlikely: 2L

// Test for Microsoft __declspec attributes

#define DECLSPEC(x) x: __has_declspec_attribute(x)

// CHECK: uuid: 1
// CHECK: __uuid__: 1
DECLSPEC(uuid)
DECLSPEC(__uuid__)

// CHECK: fallthrough: 0
DECLSPEC(fallthrough)
