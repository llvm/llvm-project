// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

_Alignas(int) struct c1; // expected-warning {{'_Alignas' attribute ignored}}
alignas(int) struct c1; // expected-warning {{'alignas' attribute ignored}}
