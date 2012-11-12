// RUN: %clang_cc1 -fsyntax-only -trigraphs -Wtrigraphs -verify %s

??=pragma // expected-warning {{trigraph converted to '#' character}}

int a = '??/0'; // expected-warning {{trigraph converted to '\' character}}

int b = 1 ??' 0; // expected-warning {{trigraph converted to '^' character}}

int c ??(1]; // expected-warning {{trigraph converted to '[' character}}

int d [1??); // expected-warning {{trigraph converted to ']' character}}

int e = 1 ??! 0; // expected-warning {{trigraph converted to '|' character}}

void f() ??<} // expected-warning {{trigraph converted to '{' character}}

void g() {??> // expected-warning {{trigraph converted to '}' character}}

int h = ??- 0; // expected-warning {{trigraph converted to '~' character}}
