// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++17  %s

// expected-no-diagnostics

@interface I;
@end

@class C;

static_assert(__is_same(__add_pointer(id), id*));
static_assert(__is_same(__add_pointer(I), I*));

static_assert(__is_same(__remove_pointer(C*), C));
static_assert(!__is_same(__remove_pointer(id), id));
static_assert(__is_same(__remove_pointer(id*), id));
static_assert(__is_same(__remove_pointer(__add_pointer(id)), id));
static_assert(__is_same(__add_pointer(__remove_pointer(id)), id));
