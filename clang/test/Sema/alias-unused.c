// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wunneeded-internal-declaration -x c -verify %s
// expected-no-diagnostics
static int f() { return 42; }
int g() __attribute__((alias("f")));

static int foo [] = { 42, 0xDEAD };
extern typeof(foo) bar __attribute__((unused, alias("foo")));
