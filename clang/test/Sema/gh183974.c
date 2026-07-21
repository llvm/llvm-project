// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

__block int x; // expected-error {{'__block' is not allowed on a nonlocal variable}}

int x;
