// RUN: %check_clang_tidy -std=c++17-or-later %s readability-redundant-inline-specifier %t -- -config="{CheckOptions: {readability-redundant-inline-specifier.DiagnoseStaticInline: 'false'}}"

// With DiagnoseStaticInline disabled, 'static inline' is left alone.
static inline int fn0(int i)
{
    return i - 1;
}

static inline int STATIC_INLINE_VAR = 42;

// Declarations that are redundantly inline for another reason are still
// reported.
constexpr inline void fn1() {}
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: function 'fn1' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: constexpr void fn1() {}
