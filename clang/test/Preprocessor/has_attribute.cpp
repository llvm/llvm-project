// RUN: %clang_cc1 -triple i386-unknown-unknown -fms-compatibility -std=c++11 -E -P %s -o - | FileCheck %s --check-prefixes=CHECK,ITANIUM --implicit-check-not=:
// RUN: %clang_cc1 -triple i386-windows -fms-compatibility -std=c++11 -E -P %s -o - | FileCheck %s --check-prefixes=CHECK,WINDOWS --implicit-check-not=:

#define CXX11(x) x: __has_cpp_attribute(x)

// CHECK: clang::fallthrough: 1
CXX11(clang::fallthrough)

// CHECK: selectany: 0
CXX11(selectany)

// The attribute name can be bracketed with double underscores.
// CHECK: clang::__fallthrough__: 1
CXX11(clang::__fallthrough__)

// The scope cannot be bracketed with double underscores unless it is
// for gnu or clang.
// CHECK: __gsl__::suppress: 0
CXX11(__gsl__::suppress)

// CHECK: _Clang::fallthrough: 1
CXX11(_Clang::fallthrough)

// CHECK: __nodiscard__: 201907L
CXX11(__nodiscard__)

// CHECK: warn_unused_result: 0
CXX11(warn_unused_result)

// CHECK: gnu::warn_unused_result: 1
CXX11(gnu::warn_unused_result)

// CHECK: clang::warn_unused_result: 1
CXX11(clang::warn_unused_result)

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
CXX11(msvc::no_unique_address)
CXX11(nodiscard)
CXX11(noreturn)
CXX11(unlikely)
// FIXME(201806L) CHECK: assert: 0
// CHECK: carries_dependency: 200809L
// CHECK: deprecated: 201309L
// FIXME(201806L) CHECK: ensures: 0
// FIXME(201806L) CHECK: expects: 0
// CHECK: fallthrough: 201603L
// CHECK: likely: 201803L
// CHECK: maybe_unused: 201603L
// ITANIUM: no_unique_address: 201803L
// WINDOWS: no_unique_address: 0 
// ITANIUM: msvc::no_unique_address: 0
// WINDOWS: msvc::no_unique_address: 201803L
// CHECK: nodiscard: 201907L
// CHECK: noreturn: 200809L
// CHECK: unlikely: 201803L

namespace PR48462 {
// Test that macro expansion of the builtin argument works.
#define C clang
#define F fallthrough
#define CF clang::fallthrough

#if __has_cpp_attribute(F)
int has_fallthrough;
#endif
// CHECK: int has_fallthrough;

#if __has_cpp_attribute(C::F)
int has_clang_falthrough_1;
#endif
// CHECK: int has_clang_falthrough_1;

#if __has_cpp_attribute(clang::F)
int has_clang_falthrough_2;
#endif
// CHECK: int has_clang_falthrough_2;

#if __has_cpp_attribute(C::fallthrough)
int has_clang_falthrough_3;
#endif
// CHECK: int has_clang_falthrough_3;

#if __has_cpp_attribute(CF)
int has_clang_falthrough_4;
#endif
// CHECK: int has_clang_falthrough_4;

#define FUNCLIKE1(x) clang::x
#if __has_cpp_attribute(FUNCLIKE1(fallthrough))
int funclike_1;
#endif
// CHECK: int funclike_1;

#define FUNCLIKE2(x) _Clang::x
#if __has_cpp_attribute(FUNCLIKE2(fallthrough))
int funclike_2;
#endif
// CHECK: int funclike_2;
}

// Test for Microsoft __declspec attributes

#define DECLSPEC(x) x: __has_declspec_attribute(x)

// CHECK: uuid: 1
// CHECK: __uuid__: 1
DECLSPEC(uuid)
DECLSPEC(__uuid__)

// CHECK: fallthrough: 0
DECLSPEC(fallthrough)

namespace PR48462 {
// Test that macro expansion of the builtin argument works.
#define U uuid

#if __has_declspec_attribute(U)
int has_uuid;
#endif
// CHECK: int has_uuid;
}
