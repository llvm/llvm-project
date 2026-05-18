// RUN: %clang_cc1 %s -triple=i686-apple-darwin9 -verify -DVERIFY
// RUN: %clang_cc1 %s -triple=i686-apple-darwin9 -fms-extensions -DMS -verify -DVERIFY

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
    !__has_builtin(__builtin_FILE_NAME) || \
    !__has_builtin(__builtin_FUNCTION) || \
    !__has_builtin(__builtin_COLUMN) || \
    !__has_builtin(__array_rank) || \
    !__has_builtin(__underlying_type) || \
    !__has_builtin(__is_trivial) || \
    !__has_builtin(__is_same_as) || \
    !__has_builtin(__has_unique_object_representations) || \
    !__has_builtin(__is_trivially_equality_comparable) || \
    !__has_builtin(__reference_constructs_from_temporary) || \
    !__has_builtin(__reference_binds_to_temporary) || \
    !__has_builtin(__reference_converts_from_temporary)
#error Clang should have these
#endif

#ifdef MS
#if !__has_builtin(__builtin_FUNCSIG)
#error Clang should have this
#endif
#elif __has_builtin(__builtin_FUNCSIG)
#error Clang should not have this without '-fms-extensions'
#endif

// This is a C-only builtin.
#if __has_builtin(__builtin_types_compatible_p)
#error Clang should not have this in C++ mode
#endif

#if __has_builtin(__builtin_insanity)
#error Clang should not have this
#endif

// Check __has_constexpr_builtin
#if  !__has_constexpr_builtin(__builtin_fmax) || \
     !__has_constexpr_builtin(__builtin_fmin) || \
     !__has_constexpr_builtin(__builtin_fmaximum_num) || \
     !__has_constexpr_builtin(__builtin_fminimum_num)
#error Clang should have these constexpr builtins
#endif

#if !__has_constexpr_builtin(__builtin_convertvector)
#error Clang should have these constexpr builtins
#endif

#if !__has_constexpr_builtin(__builtin_shufflevector)
#error Clang should have these constexpr builtins
#endif

#if  __has_constexpr_builtin(__builtin_cbrt)
#error This builtin should not be constexpr in Clang
#endif

#if  __has_constexpr_builtin(__builtin_insanity)
#error This is not a builtin in Clang
#endif

// expected-error@+1 {{missing '(' after '__has_constexpr_builtin'}} expected-error@+1 {{expected value}}
#if __has_constexpr_builtin
#endif

// expected-error@+1 {{builtin feature check macro requires a parenthesized identifier}}
#if  __has_constexpr_builtin("__builtin_fmax")
#endif

// expected-error@+1 {{too many arguments}}
#if __has_constexpr_builtin(__builtin_fmax, __builtin_fmin)
#endif
