// RUN: %clang_cc1 -triple %itanium_abi_triple -Wunused -x c -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -Wunused -x c++ -verify=expected,cxx %s

#ifdef __cplusplus
extern "C" {
#endif
static int f(void) { return 42; }
int g(void) __attribute__((alias("f")));

static int foo [] = { 42, 0xDEAD }; // cxx-warning{{variable 'foo' is not needed and will not be emitted}}
extern typeof(foo) bar __attribute__((unused, alias("foo")));

static int (*resolver(void))(void) { return f; } // cxx-warning{{unused function 'resolver'}}
int ifunc(void) __attribute__((ifunc("resolver")));

static int __attribute__((overloadable)) f0(int x) { return x; }
static float __attribute__((overloadable)) f0(float x) { return x; } // expected-warning{{unused function 'f0'}}
int g0(void) __attribute__((alias("_ZL2f0i")));

#ifdef __cplusplus
static int f1() { return 42; }
int g1(void) __attribute__((alias("_ZL2f1v")));
}

/// We demangle alias/ifunc target and mark all found functions as used.

static int f2(int) { return 42; } // cxx-warning{{unused function 'f2'}}
static int f2() { return 42; }
int g2() __attribute__((alias("_ZL2f2v")));

static int (*resolver1())() { return f; } // cxx-warning{{unused function 'resolver1'}}
static int (*resolver1(int))() { return f; }
int ifunc1() __attribute__((ifunc("_ZL9resolver1i")));

/// We should report "unused function" for f3(int).
namespace ns {
static int f3(int) { return 42; } // cxx-warning{{unused function 'f3'}}
static int f3() { return 42; } // cxx-warning{{unused function 'f3'}}
int g3() __attribute__((alias("_ZN2nsL2f3Ev")));
}

template <class T>
static void *f4(T) { return nullptr; }
static void *f4() { return nullptr; } // cxx-warning{{unused function 'f4'}}
extern void g4() __attribute__((ifunc("_ZL2f4IiEPvT_")));
void *use3 = f4(0);
#endif
