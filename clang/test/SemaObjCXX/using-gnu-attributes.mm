// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-gc -verify %s

// GNU attributes in the type-id of an alias-declaration apply to the alias like
// they do for the equivalent typedef.

@class X;

typedef __attribute__((objc_ownership(weak))) int *TP; // expected-warning {{'__weak' only applies to Objective-C object or block pointer types; type here is 'int *'}}
using UP = __attribute__((objc_ownership(weak))) int *; // expected-warning {{'__weak' only applies to Objective-C object or block pointer types; type here is 'int *'}}
typedef __attribute__((objc_gc(weak))) int *TQ;
using UQ = __attribute__((objc_gc(weak))) int *;
typedef __attribute__((objc_ownership(strong))) X *(^TB)(void);
using UB = __attribute__((objc_ownership(strong))) X *(^)(void);
static_assert(__is_same(TP, UP) && __is_same(TQ, UQ) && __is_same(TB, UB), "");
