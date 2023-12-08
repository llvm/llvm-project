// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

@class X;

static_assert(__is_same(__remove_pointer(X *), X), "");
static_assert(__is_same(__remove_pointer(id), id), "");
