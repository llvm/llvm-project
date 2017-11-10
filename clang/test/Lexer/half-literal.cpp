// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
float a = 1.0h; // expected-error{{invalid suffix 'h' on floating constant}}
float b = 1.0H; // expected-error{{invalid suffix 'H' on floating constant}}

_Float16 c = 1.f166; // expected-error{{invalid suffix 'f166' on floating constant}}
_Float16 d = 1.f1;   // expected-error{{invalid suffix 'f1' on floating constant}}
