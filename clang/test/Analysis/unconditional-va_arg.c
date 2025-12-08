// RUN: %clang_analyze_cc1 -triple hexagon-unknown-linux -verify %s \
// RUN:   -analyzer-checker=core,optin.core.UnconditionalVAArg \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -verify %s \
// RUN:   -analyzer-checker=core,optin.core.UnconditionalVAArg \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text

#include "Inputs/system-header-simulator-for-valist.h"
int printf(const char *format, ...);
void abort(void);

void log_message(const char *msg, ...) {
  // This artifical but plausible example function expects that the list of
  // variadic arguments is terminated by a NULL pointer. Although careful use
  // of this function is well-defined, passing no variadic arguments would
  // trigger undefined behavior, so it may be better to choose a different way
  // to mark the count of variadic arguments.
  va_list va;
  const char *arg;
  printf("%s\n", msg);
  va_start(va, msg);
  while ((arg = va_arg(va, const char *))) {
    // expected-warning@-1 {{Unconditional use of va_arg()}}
    // expected-note@-2 {{Calls to 'log_message' always reach this va_arg() expression, so calling 'log_message' with no variadic arguments would be undefined behavior}}
    printf(" * %s\n", arg);
  }
  va_end(va);
}

void simple(int fst, ...) {
  // This function is not called in this test file.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'simple' always reach this va_arg() expression, so calling 'simple' with no variadic arguments would be undefined behavior}}
  va_end(va);
}

void used_with_varargs(int fst, ...) {
  // This function is identical to simple(), but it is used and the call passes
  // several variadic arguments.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'used_with_varargs' always reach this va_arg() expression, so calling 'used_with_varargs' with no variadic arguments would be undefined behavior}}
  va_end(va);
}

void caller1(void) {
  used_with_varargs(1, 2, 3, 4);
}

void used_without_varargs(int fst, ...) {
  // This function is identical to simple(), but it is used and the call passes
  // no variadic arguments.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'used_without_varargs' always reach this va_arg() expression, so calling 'used_without_varargs' with no variadic arguments would be undefined behavior}}
  // FIXME: Here the checker should mention that this function is _actually_
  // called with no variadic arguments.
  va_end(va);
}

void caller2(void) {
  used_without_varargs(1);
}

void multiple_va_arg_calls(int fst, ...) {
  // This function is similar to simple() but calls va_arg() multiple times. To
  // avoid spamming the user, only the first use is reported.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'multiple_va_arg_calls' always reach this va_arg() expression, so calling 'multiple_va_arg_calls' with no variadic arguments would be undefined behavior}}
  (void)va_arg(va, int); // no-warning
  (void)va_arg(va, int); // no-warning
  va_end(va);
}


void has_conditional_return(int fst, ...) {
  // The va_arg call is not always executed, so don't report this function.
  // Actually understanding the conditional logic is infeasible, so the checker
  // should accept any conditional logic that sometimes avoids the va_arg().
  va_list va;
  if (fst < 0)
    return;
  va_start(va, fst);
  (void)va_arg(va, int); // no-warning
  va_end(va);
}

void has_conditional_logic(int fst, ...) {
  // As static analyzer only follows one execution path, it cannot see that
  // va_arg() is always executed in this function. Theoretically it would be
  // better to produce a report here, but implementing this would be a very
  // ineffective use of development effort.
  va_list va;
  if (fst < 0)
    fst = -fst;
  va_start(va, fst);
  (void)va_arg(va, int); // no-warning
  va_end(va);
}

void caller_has_conditional_logic(int fst, ...) {
  // This function is identical to simple(), but it is used by a function that
  // performs a state split before the call.
  // This test validates that state splits _before_ the variadic call (which
  // are irrelevant) do not influence the report creation. This is basically a
  // workaround for the fact that not all functions are entrypoints and before
  // reaching "our" variadic function the analyzer may go through arbitrary
  // irrelevant code.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'caller_has_conditional_logic' always reach this va_arg() expression, so calling 'caller_has_conditional_logic' with no variadic arguments would be undefined behavior}}
  va_end(va);
}

void caller_with_conditional_logic(int flag) {
  if (flag)
    caller_has_conditional_logic(1, 2, 3);
}


void validate_argument(int fst) {
  if (fst <= 0) {
    printf("Negative first argument is illegal: %d!\n", fst);
    abort();
  }
}

void noreturn_in_function(int fst, ...) {
  // This function is very similar to simple(), but calls a function which may
  // call the noreturn function abort() before reaching the va_arg() call, so
  // the checker doesn't produce a warning. This is important e.g. for cases
  // when a function uses assertions to mark preconditions.
  va_list va;
  validate_argument(fst);
  va_start(va, fst);
  (void)va_arg(va, int); // no-warning
  va_end(va);
}

void print_negative(int fst) {
  if (fst <= 0) {
    printf("First argument is negative: %d!\n", fst);
  }
}

void conditional_in_function(int fst, ...) {
  // This function is very similar to noreturn_in_function(), but the called
  // function always returns. However, the analyzer doesn't see this difference
  // (because it follows a _single_ execution path and cannot see the
  // alternative) so the checker behaves as in noreturn_in_function.
  va_list va;
  print_negative(fst);
  va_start(va, fst);
  (void)va_arg(va, int); // no-warning
  va_end(va);
}

void defined_in_different_tu(int);

void unknown_call(int fst, ...) {
  // This function is very similar to noreturn_in_function() or
  // conditional_in_function() but the definition of the called function is not
  // visible for the analyzer. In theis situation the analyzer ignores the call
  // and reports the va_arg() expression that unconditionally appears after it.
  // This would be incorrect behavior only in a situation when the called
  // function can return _and_ can be noreturn which should be fairly rare.
  va_list va;
  defined_in_different_tu(fst);
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'unknown_call' always reach this va_arg() expression, so calling 'unknown_call' with no variadic arguments would be undefined behavior}}
  va_end(va);
}
