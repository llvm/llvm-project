// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -fopenmp -verify %s

void fr(int *);

// A check in an initializer leaves before the variable is initialized, so a
// handler later in the variable's scope bypasses the initialization.
int cleanup_init(int __ob_trap a) {
  int __attribute__((cleanup(fr))) x = a + 1; // expected-error {{cannot jump from this overflow-checked operation to its overflow handler label}} \
                                              // expected-note {{jump bypasses initialization of variable with __attribute__((cleanup))}}
  return x;
__overflow:
  return -1;
}

// A handler outside the variable's scope is fine, and cleanups that have not
// started do not run.
int cleanup_init_outer(int __ob_trap a) {
  {
    int __attribute__((cleanup(fr))) x = a + 1;
    return x;
  }
__overflow:
  return -1;
}

// So is a handler within the initializer itself.
int handler_in_initializer(int __ob_trap a) {
  int __attribute__((cleanup(fr))) x = ({
    __label__ __overflow;
    int r = a + 1;
    if (0) {
    __overflow:
      r = 0;
    }
    r;
  });
  return x;
}

// Variable length array sizes trap, so they never jump past the allocation.
int vla_size(int __ob_trap n) {
  int v[n + 1];
  v[0] = 0;
  return v[0];
__overflow:
  return -1;
}

// Operands that are not evaluated never branch.
int unevaluated(int __ob_trap a, int n) {
  int s = sizeof(a + 1) + _Generic(0, int: 1, long: a + 1) +
          __builtin_choose_expr(0, a + 1, 2);
  int v[n];
  v[0] = s;
  return v[0];
__overflow:
  return -1;
}

// An OpenMP region cannot leave its structured block.
int openmp_region(int __ob_trap a, int n) {
  int v[n];
  #pragma omp parallel
  { a = a + 1; }
  v[0] = a;
  return v[0];
__overflow:
  return -1;
}
