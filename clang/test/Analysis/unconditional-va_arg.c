// RUN: %clang_analyze_cc1 -triple hexagon-unknown-linux -verify %s \
// RUN:   -analyzer-checker=core,optin.core.UnconditionalVAArg \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -verify %s \
// RUN:   -analyzer-checker=core,optin.core.UnconditionalVAArg \
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


void never_called(int fst, ...) {
  // This simple test function is not called in this test file.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'never_called' always reach this va_arg() expression, so calling 'never_called' with no variadic arguments would be undefined behavior}}
  va_end(va);
}


void called_with_varargs(int fst, ...) {
  // This function is identical to never_called(), but it is called by another
  // function (which will serve as the entrypoint) and the call passes several
  // variadic arguments. The diagnostic is still raised.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'called_with_varargs' always reach this va_arg() expression, so calling 'called_with_varargs' with no variadic arguments would be undefined behavior}}
  va_end(va);
}

void caller1(void) {
  called_with_varargs(1, 2, 3, 4);
}


void called_without_varargs(int fst, ...) {
  // This function is identical to never_called(), but it is called by another
  // function (which will serve as the entrypoint) and the call passes no
  // variadic arguments. The diagnostic is still raised.
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'called_without_varargs' always reach this va_arg() expression, so calling 'called_without_varargs' with no variadic arguments would be undefined behavior}}
  // FIXME: Here the checker should mention that this function is _actually_
  // called with no variadic arguments.
  va_end(va);
}

void caller2(void) {
  called_without_varargs(1);
}


void multiple_va_arg_calls(int fst, ...) {
  // This function function unconditionally calls va_arg() multiple times. To
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


#define IS_FEATURE_ENABLED 1
void has_constant_condition(int fst, ...) {
  // This checker doesn't have special logic for recognizing and ignoring
  // conditionals that can be resolved at compile time. This feature could be
  // added in the future.
  va_list va;
  if (IS_FEATURE_ENABLED) {
    va_start(va, fst);
    (void)va_arg(va, int); // no-warning
    va_end(va);
  }
}



void caller_has_conditional_logic(int fst, ...) {
  // This function is identical to never_called(), but it is used by a function
  // that performs a state split before the call.
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


int has_shortcircuiting_operator(int fst, ...) {
  // In addition to 'if' and similar statements, ternary operators and
  // shortcircuiting logical operators can also ensure that the use of va_arg()
  // is not unconditional. (These all behave identically in the BranchCondition
  // callback, so this simple testcase is mostly for documentation purposes.)
  va_list va;
  int x;
  va_start(va, fst);
  x = fst || va_arg(va, int); // no-warning
  va_end(va);
  return x;
}


void validate_argument(int fst) {
  if (fst <= 0) {
    printf("Negative first argument is illegal: %d!\n", fst);
    abort();
  }
}

void noreturn_in_function(int fst, ...) {
  // This function calls a function which may call the noreturn function
  // abort() before reaching the va_arg() call, so the checker doesn't produce
  // a warning. This behavior is important e.g. for cases when a function uses
  // assertions to mark preconditions.
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


void trivial_variadic(int fst, ...) {
  // Do nothing.
}

void calls_other_variadic(int fst, ...) {
  // As the 'HasUnconditionalPath' can "remember" only one variadic function,
  // I would expect that the presence of 'trivial_variadic' would prevent the
  // checker from reporting the unconditional use of va_arg() in this function.
  // However, for some unclear reason the checker is still able to produce this
  // (true positive) report.
  va_list va;
  trivial_variadic(fst, 2);
  va_start(va, fst);
  (void)va_arg(va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'calls_other_variadic' always reach this va_arg() expression, so calling 'calls_other_variadic' with no variadic arguments would be undefined behavior}}
  va_end(va);
}

int get_int_from_va_list(va_list *va) {
  // This checker can report va_arg() expressions that are unconditionally
  // reached from a variadic function, even if they are located in a different
  // function. In this case the originating variadic function is marked with a
  // note (to make it easier to find).
  return va_arg(*va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'uses_wrapped_va_arg' always reach this va_arg() expression, so calling 'uses_wrapped_va_arg' with no variadic arguments would be undefined behavior}}
}

void uses_wrapped_va_arg(int fst, ...) {
  // expected-note@-1 {{Variadic function 'uses_wrapped_va_arg' is defined here}}
  va_list va;
  va_start(va, fst);
  (void)get_int_from_va_list(&va);
  va_end(va);
}

int standalone_non_variadic(va_list *va) {
  // A va_arg() expression is only reported if it is unconditionally reached
  // from the beginning of a _variadic_ function. Functions that use va_arg()
  // on a va_list obtained from other sources (e.g. as an argument) are
  // presumed to be small helper subroutines of a complex variadic function, so
  // we do not report cases where the va_arg() is unconditionally reached from
  // the beginning of a _non-variadic_ function.
  return va_arg(*va, int); //no-warning
}

int variadic_but_also_takes_va_list(va_list *va, ...) {
  // The checker just looks for va_arg() calls without checking the origin of
  // their argument, so in this (very artifical) example it produces a
  // result that is arguably a false positive.
  return va_arg(*va, int); // expected-warning{{Unconditional use of va_arg()}}
  // expected-note@-1 {{Calls to 'variadic_but_also_takes_va_list' always reach this va_arg() expression, so calling 'variadic_but_also_takes_va_list' with no variadic arguments would be undefined behavior}}
}
