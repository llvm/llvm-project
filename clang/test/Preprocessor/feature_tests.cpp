// RUN: %clang_cc1 %s -triple=i686-apple-darwin9 -verify -DVERIFY
// expected-no-diagnostics

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
    !__has_builtin(__array_rank) || \
    !__has_builtin(__underlying_type) || \
    !__has_builtin(__is_trivial) || \
    !__has_builtin(__is_same_as) || \
    !__has_builtin(__has_unique_object_representations)
#error Clang should have these
#endif

// This is a C-only builtin.
#if __has_builtin(__builtin_types_compatible_p)
#error Clang should not have this in C++ mode
#endif

#if __has_builtin(__builtin_insanity)
#error Clang should not have this
#endif
