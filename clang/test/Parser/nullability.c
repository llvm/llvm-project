// RUN: %clang_cc1 -fsyntax-only -std=c99 -Wno-nullability-declspec -pedantic %s -verify

_Nonnull int *ptr; // expected-warning{{type nullability specifier '_Nonnull' is a Clang extension}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
_Nonnull int *ptr2; // no-warning
#pragma clang diagnostic pop

#if !__has_feature(nullability)
#  error Nullability should always be supported
#endif

#if !__has_feature(nullability_on_arrays)
#  error Nullability on array parameters should always be supported
#endif

#if !__has_extension(nullability)
#  error Nullability should always be supported as an extension
#endif

#if !__has_extension(nullability_on_arrays)
#  error Nullability on array parameters should always be supported as an extension
#endif
