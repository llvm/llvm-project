// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -Wunreachable-code-aggressive -Wuninitialized -Wsometimes-uninitialized -Wconditional-uninitialized -verify %s

// Flow-sensitive warnings see the branch from an overflow checked operation to
// its handler label.

void fr(int *);

int function_handler(int __ob_trap a) {
  a = a + 1;
  return a;
__overflow:
  return -1;
}

int local_handler(int __ob_trap a) {
  {
    __label__ __overflow;
    a = a * 2;
    return a;
  __overflow:
    return -2;
  }
}

// The handler path skips the store to r.
int uninit_on_handler_path(int __ob_trap a) {
  int r; // expected-note {{initialize the variable 'r' to silence this warning}}
  a = a + 1;
  r = 0;
  return a;
__overflow:
  return r; // expected-warning {{variable 'r' is uninitialized when used here}}
}

// r is initialized on the continuation but not on the handler path.
int sometimes_uninit(int __ob_trap a) {
  int r; // expected-note {{initialize the variable 'r' to silence this warning}}
  a = a + 1; // expected-warning {{variable 'r' is used uninitialized whenever an overflow check fails}}
  r = 0;
__overflow:
  return r + a; // expected-note {{uninitialized use occurs here}}
}

// Uninitialized only through an earlier branch: the overflow branch is not
// the one to blame.
int maybe_uninit(int __ob_trap a, int c) {
  int r; // expected-note {{initialize the variable 'r' to silence this warning}}
  if (c)
    r = 1;
  a = a + 1;
  return a;
__overflow:
  return r; // expected-warning {{variable 'r' may be uninitialized when used here}}
}

// The check happens after r is assigned, so r is initialized on both paths.
int init_before_check(int __ob_trap a) {
  int r = 0;
  a = a + 1;
  return a;
__overflow:
  return r;
}

// Leaving a scope with a cleanup on the way to the handler.
int cleanup_scope(int __ob_trap a) {
  {
    int __attribute__((cleanup(fr))) x = 0;
    a = a + x;
  }
  return a;
__overflow:
  return -1;
}

// A label before the operation is reached backward.
int backward(int __ob_trap a, int n) {
  int r = 1;
__overflow:
  if (n-- > 0)
    r = a * r;
  return r;
}

// Without a check that can reach it, the label is still unreachable.
int unchecked(int a) {
  a = a + 1;
  return a;
__overflow:
  return -1; // expected-warning {{'return' will never be executed}}
}

// Wrap types never branch.
int wrap(int __ob_wrap a) {
  a = a + 1;
  return a;
__overflow:
  return -1; // expected-warning {{'return' will never be executed}}
}
