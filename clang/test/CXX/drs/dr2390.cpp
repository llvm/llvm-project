// RUN: %clang_cc1 -E -P %s -o - | FileCheck %s

// dr2390: yes

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
int has_clang_fallthrough_1;
#endif
// CHECK: int has_clang_fallthrough_1;

#if __has_cpp_attribute(clang::F)
int has_clang_fallthrough_2;
#endif
// CHECK: int has_clang_fallthrough_2;

#if __has_cpp_attribute(C::fallthrough)
int has_clang_fallthrough_3;
#endif
// CHECK: int has_clang_fallthrough_3;

#if __has_cpp_attribute(CF)
int has_clang_fallthrough_4;
#endif
// CHECK: int has_clang_fallthrough_4;

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
} // namespace PR48462
