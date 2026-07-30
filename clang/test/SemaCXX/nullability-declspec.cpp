// RUN: %clang_cc1 -fblocks -Werror=nullability-declspec -verify %s

struct X { };

_Nullable int *ip1; // expected-error{{nullability specifier '_Nullable' cannot be applied to non-pointer type 'int'; did you mean to apply the specifier to the pointer?}}
_Nullable int (*fp1)(int); // expected-error{{nullability specifier '_Nullable' cannot be applied to non-pointer type 'int'; did you mean to apply the specifier to the function pointer?}}
_Nonnull int (^bp1)(int); // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'; did you mean to apply the specifier to the block pointer?}}
_Nonnull int X::*pmd1; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'; did you mean to apply the specifier to the member pointer?}}
_Nonnull int (X::*pmf1)(int); // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'; did you mean to apply the specifier to the member function pointer?}}
