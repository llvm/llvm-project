// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o %t

// Note-LABEL: define CLANG_GENERATE_KNOWN_GOOD and compile to generate code
// that makes all of the defaulted arguments explicit. The resulting
// byte code should be identical to the compilation without
// CLANG_GENERATE_KNOWN_GOOD.
#ifdef CLANG_GENERATE_KNOWN_GOOD
#  define DEFARG(...) __VA_ARGS__
#else
#  define DEFARG(...)
#endif

extern int x;
struct S { float x; float y; } s;
double _Complex c;

void f(int i = 0, int j = 1, int k = x, struct S t = s, double _Complex d = c);

void g() {
  f(0, 1, x, s DEFARG(, c));
  f(0, 1, x DEFARG(, s, c));
  f(0, 1 DEFARG(, x, s, c));
  f(0 DEFARG(, 1, x, s, c));
  f(DEFARG(0, 1, x, s, c));
}
