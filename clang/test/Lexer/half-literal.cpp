// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
float a = 1.0h; // expected-error{{invalid suffix 'h' on floating constant}}
float b = 1.0H; // expected-error{{invalid suffix 'H' on floating constant}}
