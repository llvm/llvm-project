// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 -std=c++17 -emit-llvm-only -verify=use %s
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 -std=c++17 -emit-llvm-only -verify=ret -DRET %s
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 -std=c++17 -emit-llvm-only -verify=param -DPARAM %s
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-128 -std=c++17 -emit-llvm-only -verify=ld128 %s

// ld128-no-diagnostics

// Unused inline definitions, such as those in <cmath>, are not emitted.
inline long double unused_fabs(long double x) { return __builtin_fabsl(x); }

long double g = 1.5L;
struct S { long double x; } gs = {2.5L};

// Only the layout is needed here.
static_assert(sizeof(long double) == 16 && alignof(long double) == 16);
long double *addr() { return &g; }
void copy(S *d) { *d = gs; }
void pass(S s) {}

#if RET
// An address-taken declaration needs an ARM64EC exit thunk.
// ret-error@+1 {{x87 'long double' values are not supported on target 'arm64ec-w64-windows-gnu'}}
long double ret();
void *take() { return (void *)&ret; }
#elif PARAM
// param-error@+1 {{x87 'long double' values are not supported}}
void param(long double) {}
#else
typedef long double v2ld __attribute__((vector_size(32)));

void use(double d, _Complex long double *c, _Atomic long double *a, v2ld *v) {
  // use-error@+1 {{x87 'long double' values are not supported}}
  d = g;
  // use-error@+1 {{x87 'long double' values are not supported}}
  gs.x = d;
  // use-error@+1 2 {{x87 'long double' values are not supported}}
  d = g < gs.x;
  // use-error@+1 {{x87 'long double' values are not supported}}
  d = __real__ *c;
  // use-error@+1 {{x87 'long double' values are not supported}}
  *a += 1;
  // use-error@+1 {{x87 'long double' values are not supported}}
  *v = *v + *v;
  // use-error@+1 {{x87 'long double' values are not supported}}
  unused_fabs(0);
}
#endif
