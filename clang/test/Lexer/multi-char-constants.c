// RUN: %clang_cc1 -fsyntax-only -verify -Wfour-char-constants -pedantic-errors %s

int x = 'ab'; // expected-warning {{multi-character character constant}}
int y = 'abcd'; // expected-warning {{multi-character character constant}}
