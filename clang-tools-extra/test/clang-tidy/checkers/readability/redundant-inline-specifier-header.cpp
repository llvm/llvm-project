// RUN: %check_clang_tidy -std=c++17-or-later %s -assume-filename=redundant-inline-specifier-header.hpp readability-redundant-inline-specifier %t

// OK -- not redundant in a header file.
static inline int fn0(int i)
{
    return i - 1;
}

static inline int STATIC_INLINE_VAR = 42;

// Redundant in header file as well as implementation file.
constexpr inline void fn1() {}
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: function 'fn1' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: constexpr void fn1() {}
