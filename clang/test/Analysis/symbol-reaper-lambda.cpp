// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

template <typename... Ts>
void escape(Ts&...);
struct Dummy {};

int strange(Dummy param) {
  Dummy local_pre_lambda;
  int ref_captured = 0;

  // LambdaExpr is modeled as lazyCompoundVal of tempRegion, that contains
  // all captures. In this instance, this region contains a pointer/reference
  // to ref_captured variable.
  auto fn = [&] {
    escape(param, local_pre_lambda);
    return ref_captured; // no-warning: The value is not garbage.
  };

  int local_defined_after_lambda; // Unused, but necessary! Important that it's before the call.

  // The ref_captured binding should not be pruned at this point, as it is still
  // accessed via reference captured in operator() of fn.
  return fn();
}

