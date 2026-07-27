// RUN: %clang_cc1 -verify %s

template <class> void foo(int = 0;
// expected-error@-1 {{expected ')'}}
// expected-note@-2 {{to match this '('}}

void bar();

#include "dadfoksbdsdldfafszcf.h"
// expected-error@-1 {{file not found}}

void baz() { foo<int>(); }
