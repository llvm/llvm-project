// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

typedef void(fatal_fun)() __attribute__((__noreturn__));
fatal_fun* fatal_fptr;
void fatal_decl() __attribute__((__noreturn__));

int rng();

/// This code calls a [[noreturn]] function pointer, which used to be handled
/// inconsistently between AST builder and CSA.
/// In the result, CSA produces a path where this function returns non-0.
int return_zero_or_abort_by_fnptr() {
  if (rng()) fatal_fptr();
  return 0;
}

/// This function calls a [[noreturn]] function.
/// If it does return, it always returns 0.
int return_zero_or_abort_by_direct_fun() {
  if (rng()) fatal_decl();
  return 0;
}

/// Trigger a division by zero issue depending on the return value
/// of the called functions.
int caller() {
  int x = 0;
  // The following if branches must never be taken.
  if (return_zero_or_abort_by_fnptr())
    return 1 / x; // no-warning: Dead code.
  if (return_zero_or_abort_by_direct_fun())
    return 1 / x; // no-warning: Dead code.

  // Make sure the warning is still reported when viable.
  return 1 / x; // expected-warning {{Division by zero}}
}
