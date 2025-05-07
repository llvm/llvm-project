

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

void funcAttr1(int * __counted_by_or_null(len) p, int len) __attribute__((nonnull(1))); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}
void funcAttr2(int * __counted_by_or_null(len) p, int len) __attribute__((nonnull)); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}
__attribute__((nonnull)) void funcAttr3(int * __counted_by_or_null(len) p, int len); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}
int * __counted_by_or_null(len) funcAttr4(int len) __attribute__((nonnull)); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
__attribute__((nonnull)) int * __counted_by_or_null(len) funcAttr5(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
int * __attribute__((nonnull)) __counted_by_or_null(len) funcAttr6(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}

void funcAttr7(int * __sized_by_or_null(len) p, int len) __attribute__((nonnull(1))); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}
void funcAttr8(int * __sized_by_or_null(len) p, int len) __attribute__((nonnull)); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}
__attribute__((nonnull)) void funcAttr9(int * __sized_by_or_null(len) p, int len); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}
int * __sized_by_or_null(len) funcAttr10(int len) __attribute__((nonnull)); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
__attribute__((nonnull)) int * __sized_by_or_null(len) funcAttr11(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
int * __attribute__((nonnull)) __sized_by_or_null(len) funcAttr12(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}

void funcAttr13(int * __counted_by(len) p, int len) __attribute__((nonnull(1)));
void funcAttr14(int * __counted_by(len) p, int len) __attribute__((nonnull));
__attribute__((nonnull)) void funcAttr15(int * __counted_by(len) p, int len);
int * __counted_by(len) funcAttr16(int len) __attribute__((nonnull)); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
__attribute__((nonnull)) int * __counted_by(len) funcAttr17(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}
int * __attribute__((nonnull)) __counted_by(len) funcAttr18(int len); // expected-warning{{'nonnull' attribute applied to function with no pointer arguments}}

int * __counted_by_or_null(len) returnAttr1(int len) __attribute__((returns_nonnull)); // expected-error{{cannot combine '__counted_by_or_null' and 'returns_nonnull'; did you mean '__counted_by' instead?}}
__attribute__((returns_nonnull)) int * __counted_by_or_null(len) returnAttr2(int len); // expected-error{{cannot combine '__counted_by_or_null' and 'returns_nonnull'; did you mean '__counted_by' instead?}}
int * __attribute__((returns_nonnull)) __counted_by_or_null(len) returnAttr3(int len); // expected-error{{cannot combine '__counted_by_or_null' and 'returns_nonnull'; did you mean '__counted_by' instead?}}

int * __sized_by_or_null(len) returnAttr4(int len) __attribute__((returns_nonnull)); // expected-error{{cannot combine '__sized_by_or_null' and 'returns_nonnull'; did you mean '__sized_by' instead?}}
__attribute__((returns_nonnull)) int * __sized_by_or_null(len) returnAttr5(int len); // expected-error{{cannot combine '__sized_by_or_null' and 'returns_nonnull'; did you mean '__sized_by' instead?}}
int * __attribute__((returns_nonnull)) __sized_by_or_null(len) returnAttr6(int len); // expected-error{{cannot combine '__sized_by_or_null' and 'returns_nonnull'; did you mean '__sized_by' instead?}}

int * __counted_by(len) returnAttr7(int len) __attribute__((returns_nonnull));
__attribute__((returns_nonnull)) int * __counted_by(len) returnAttr8(int len);
int * __attribute__((returns_nonnull)) __counted_by(len) returnAttr9(int len);

void paramAttr1(int * __attribute((nonnull)) __counted_by_or_null(len) p, int len); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}
void paramAttr2(int * __counted_by_or_null(len) p __attribute((nonnull)), int len); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}
void paramAttr3(int * __counted_by_or_null(len) __attribute((nonnull)) p, int len); // expected-error{{cannot combine '__counted_by_or_null' and 'nonnull'; did you mean '__counted_by' instead?}}

void paramAttr4(int * __attribute((nonnull)) __sized_by_or_null(len) p, int len); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}
void paramAttr5(int * __sized_by_or_null(len) p __attribute((nonnull)), int len); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}
void paramAttr6(int * __sized_by_or_null(len) __attribute((nonnull)) p, int len); // expected-error{{cannot combine '__sized_by_or_null' and 'nonnull'; did you mean '__sized_by' instead?}}

void paramAttr7(int * __attribute((nonnull)) __counted_by(len) p, int len);
void paramAttr8(int * __counted_by(len) p __attribute((nonnull)), int len);
void paramAttr9(int * __counted_by(len) __attribute((nonnull)) p, int len);

void paramType1(int * _Nonnull __counted_by_or_null(len) p, int len); // expected-warning{{combining '__counted_by_or_null' and '_Nonnull'; did you mean '__counted_by' instead?}}
void paramType2(int * __counted_by_or_null(len) _Nonnull p, int len); // expected-warning{{combining '__counted_by_or_null' and '_Nonnull'; did you mean '__counted_by' instead?}}

void paramType3(int * _Nonnull __sized_by_or_null(len) p, int len); // expected-warning{{combining '__sized_by_or_null' and '_Nonnull'; did you mean '__sized_by' instead?}}
void paramType4(int * __sized_by_or_null(len) _Nonnull p, int len); // expected-warning{{combining '__sized_by_or_null' and '_Nonnull'; did you mean '__sized_by' instead?}}

void paramType5(int * _Nonnull __counted_by(len) p, int len);
void paramType6(int * __counted_by(len) _Nonnull p, int len);

void paramType7(int * __counted_by(4) _Nullable p); // expected-warning{{combining '__counted_by' with non-zero count (which cannot be null) and '_Nullable'; did you mean '__counted_by_or_null' instead?}}
void paramType8(int * __counted_by_or_null(4) _Nullable p);
void paramType9(int * __sized_by(4) _Nullable p); // expected-warning{{combining '__sized_by' with non-zero size (which cannot be null) and '_Nullable'; did you mean '__sized_by_or_null' instead?}}

void paramType10(int * _Nullable __counted_by(4) p); // expected-warning{{combining '__counted_by' with non-zero count (which cannot be null) and '_Nullable'; did you mean '__counted_by_or_null' instead?}}
void paramType11(int * _Nullable __counted_by_or_null(4) p);
void paramType12(int * _Nullable __sized_by(4) p); // expected-warning{{combining '__sized_by' with non-zero size (which cannot be null) and '_Nullable'; did you mean '__sized_by_or_null' instead?}}

