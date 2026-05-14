// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

__block int x; // expected-error {{__block attribute not allowed, only allowed on local variables}}

int x;
