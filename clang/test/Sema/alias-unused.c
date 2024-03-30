// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wunused -x c -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wunused -x c++ -verify %s

#ifdef __cplusplus
extern "C" {
#else
// expected-no-diagnostics
#endif
static int f(void) { return 42; }
int g(void) __attribute__((alias("f")));

static int foo [] = { 42, 0xDEAD };
extern typeof(foo) bar __attribute__((unused, alias("foo")));

static int (*resolver(void))(void) { return f; }
int ifunc(void) __attribute__((ifunc("resolver")));

#ifdef __cplusplus
}

/// We demangle alias/ifunc target and mark all found functions as used.
static int f1(int) { return 42; }
static int f1(void) { return 42; }
int g1(void) __attribute__((alias("_ZL2f1v")));

static int (*resolver1(void))(void) { return f; }
static int (*resolver1(int))(void) { return f; }
int ifunc1(void) __attribute__((ifunc("_ZL9resolver1v")));

namespace ns {
static int f2(int) { return 42; } // expected-warning{{unused function 'f2'}}
static int f2(void) { return 42; } // expected-warning{{unused function 'f2'}}
int g2(void) __attribute__((alias("_ZN2nsL2f2Ev")));
}
#endif
