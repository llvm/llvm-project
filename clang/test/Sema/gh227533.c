// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

const char a[4294967296] = {0};
int b(void) { return a[1]; }

const char c[4294967296] = "x";
int d(void) { return c[1]; }

const char e[4294967295] = {0};
int f(void) { return e[1]; }

const char g[4294967295] = "x";
int h(void) { return g[1]; }
