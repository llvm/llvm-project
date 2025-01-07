// RUN: %clang_cc1 -verify=wrong -std=c99 %s
// RUN: %clang_cc1 -verify=wrong -std=c11 %s
// RUN: %clang_cc1 -verify=cpp -std=c++11 -x c++ %s

/* WG14 N1285: No
 * Extending the lifetime of temporary objects (factored approach)
 *
 * NB: we do not properly materialize temporary expressions in situations where
 * it would be expected; that is why the "no-diagnostics" marking is named
 * "wrong". We do issue the expected diagnostic in C++ mode.
 */

// wrong-no-diagnostics

struct X { int a[5]; };
struct X f(void);

int foo(void) {
  // FIXME: This diagnostic should be issued in C11 as well (though not in C99,
  // as this paper was a breaking change between C99 and C11).
  int *p = f().a; // cpp-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  return *p;
}

