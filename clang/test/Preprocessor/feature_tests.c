// RUN: %clang_cc1 %s -triple=i686-apple-darwin9 -verify -DVERIFY
// RUN: %clang_cc1 %s -E -triple=i686-apple-darwin9
#ifndef __has_feature
#error Should have __has_feature
#endif


#if __has_feature(something_we_dont_have)
#error Bad
#endif

#if  !__has_builtin(__builtin_huge_val) || \
     !__has_builtin(__builtin_shufflevector) || \
     !__has_builtin(__builtin_convertvector) || \
     !__has_builtin(__builtin_trap) || \
     !__has_builtin(__c11_atomic_init) || \
     !__has_builtin(__builtin_launder) || \
     !__has_feature(attribute_analyzer_noreturn) || \
     !__has_feature(attribute_overloadable)
#error Clang should have these
#endif

// These are technically implemented as keywords, but __has_builtin should
// still return true.
#if !__has_builtin(__builtin_LINE) || \
    !__has_builtin(__builtin_FILE) || \
    !__has_builtin(__builtin_FUNCTION) || \
    !__has_builtin(__builtin_COLUMN) || \
    !__has_builtin(__builtin_types_compatible_p)
#error Clang should have these
#endif

// These are C++-only builtins.
#if __has_builtin(__array_rank) || \
    __has_builtin(__underlying_type) || \
    __has_builtin(__is_trivial)
#error Clang should not have these in C mode
#endif

#if __has_builtin(__builtin_insanity)
#error Clang should not have this
#endif

#if !__has_feature(__attribute_deprecated_with_message__)
#error Feature name in double underscores does not work
#endif

// Make sure we have x86 builtins only (forced with target triple).

#if !__has_builtin(__builtin_ia32_emms) || \
    __has_builtin(__builtin_altivec_abs_v4sf)
#error Broken handling of target-specific builtins
#endif

// Macro expansion does not occur in the parameter to __has_builtin,
// __has_feature, etc. (as is also expected behaviour for ordinary
// macros), so the following should not expand:

#define MY_ALIAS_BUILTIN __c11_atomic_init
#define MY_ALIAS_FEATURE attribute_overloadable

#if __has_builtin(MY_ALIAS_BUILTIN) || __has_feature(MY_ALIAS_FEATURE)
#error Alias expansion not allowed
#endif

// But deferring should expand:

#define HAS_BUILTIN(X) __has_builtin(X)
#define HAS_FEATURE(X) __has_feature(X)

#if !HAS_BUILTIN(MY_ALIAS_BUILTIN) || !HAS_FEATURE(MY_ALIAS_FEATURE)
#error Expansion should have occurred
#endif

#ifdef VERIFY
// expected-error@+1 {{builtin feature check macro requires a parenthesized identifier}}
#if __has_feature('x')
#endif

// The following are not identifiers:
_Static_assert(!__is_identifier("string"), "oops");
_Static_assert(!__is_identifier('c'), "oops");
_Static_assert(!__is_identifier(123), "oops");
_Static_assert(!__is_identifier(int), "oops");

// The following are:
_Static_assert(__is_identifier(abc /* comment */), "oops");
_Static_assert(__is_identifier /* comment */ (xyz), "oops");

// expected-error@+1 {{too few arguments}}
#if __is_identifier()
#endif

// expected-error@+1 {{too many arguments}}
#if __is_identifier(,())
#endif

// expected-error@+1 {{missing ')' after 'abc'}} 
#if __is_identifier(abc xyz) // expected-note {{to match this '('}}
#endif

// expected-error@+1 {{missing ')' after 'abc'}} 
#if __is_identifier(abc())   // expected-note {{to match this '('}}
#endif

// expected-error@+1 {{missing ')' after '.'}} 
#if __is_identifier(.abc)    // expected-note {{to match this '('}}
#endif

// expected-error@+1 {{nested parentheses not permitted in '__is_identifier'}} 
#if __is_identifier((abc))
#endif

// expected-error@+1 {{missing '(' after '__is_identifier'}} expected-error@+1 {{expected value}}
#if __is_identifier
#endif

// expected-error@+1 {{unterminated}} expected-error@+1 {{expected value}}
#if __is_identifier(
#endif

#endif
