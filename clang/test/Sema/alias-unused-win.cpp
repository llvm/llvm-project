// RUN: %clang_cc1 -triple %ms_abi_triple -Wunused -x c -verify %s
// RUN: %clang_cc1 -triple %ms_abi_triple -Wunused -verify=expected,cxx %s

#ifdef __cplusplus
extern "C" {
#endif
static int f(void) { return 42; } // cxx-warning{{unused function 'f'}}
int g(void) __attribute__((alias("f")));

static int foo [] = { 42, 0xDEAD };
extern typeof(foo) bar __attribute__((unused, alias("foo")));

static int __attribute__((overloadable)) f0(int x) { return x; } // expected-warning{{unused function 'f0'}}
static float __attribute__((overloadable)) f0(float x) { return x; } // expected-warning{{unused function 'f0'}}
int g0(void) __attribute__((alias("?f0@@YAHH@Z")));

#ifdef __cplusplus
/// https://github.com/llvm/llvm-project/issues/88593
/// We report a warning in C++ mode because the internal linkage `resolver` gets
/// mangled as it does not have a language linkage. GCC does not mangle
/// `resolver` or report a warning.
static int f1() { return 42; } // cxx-warning{{unused function 'f1'}}
int g1(void) __attribute__((alias("?f1@@YAHXZ")));
}

namespace ns {
static int f3(int) { return 42; } // cxx-warning{{unused function 'f3'}}
static int f3() { return 42; } // cxx-warning{{unused function 'f3'}}
int g3() __attribute__((alias("?f3@ns@@YAHXZ")));
}
#endif
