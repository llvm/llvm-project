// RUN: %clang_cc1 %s -triple=i686-apple-darwin9
// RUN: %clang_cc1 %s -E -triple=i686-apple-darwin9
#ifndef __has_feature
#error Should have __has_feature
#endif


#if __has_feature(something_we_dont_have)
#error Bad
#endif

#if  !__has_builtin(__builtin_huge_val) || \
     !__has_builtin(__builtin_shufflevector) || \
     !__has_builtin(__builtin_trap) || \
     !__has_builtin(__c11_atomic_init) || \
     !__has_feature(attribute_analyzer_noreturn) || \
     !__has_feature(attribute_overloadable)
#error Clang should have these
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
