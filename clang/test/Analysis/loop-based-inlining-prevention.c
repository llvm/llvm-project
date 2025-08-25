// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify=expected,default %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-functions-with-ambiguous-loops=true -verify=expected,enabled %s

// This file tests some heuristics in the engine that put functions on a
// "do not inline" list if their analyisis reaches the `analyzer-max-loop`
// limit (by default 4 iterations) in a loop. This was almost surely intended
// as memoization optimization for the "retry without inlining" fallback (if we
// had to retry once, next time don't even try inlining), but aggressively
// oversteps the "natural" scope: reaching 4 iterations on _one particular_
// execution path does not imply that each path would need "retry without
// inlining" especially if a different call receives different arguments.
//
// This heuristic significantly affects the scope/depth of the analysis (and
// therefore the execution time) because without this limitation on the
// inlining significantly more entry points would be able to exhaust their
// `max-nodes` quota. (Trivial thin wrappers around big complex functions are
// common in many projects.)
//
// Unfortunately, this arbitrary heuristic strongly relies on the current loop
// handling model and its many limitations, so improvements in loop handling
// can cause surprising slowdowns by reducing the "do not inline" blacklist.
// In the tests "FIXME-BUT-NEEDED" comments mark "problematic" (aka buggy)
// analyzer behavior which cannot be fixed without also improving the
// heuristics for (not) inlining large functions.

  int getNum(void); // Get an unknown symbolic number.

void clang_analyzer_dump(int arg);

//-----------------------------------------------------------------------------
// Simple case: inlined function never reaches `analyzer-max-loop`, so it is
// always inlined.

int inner_simple(int callIdx) {
  clang_analyzer_dump(callIdx); // expected-warning {{1 S32}}
                                // expected-warning@-1 {{2 S32}}
  return 42;
}

int outer_simple(void) {
  int x = inner_simple(1);
  int y = inner_simple(2);
  return 53 / (x - y); // expected-warning {{Division by zero}}
}

//-----------------------------------------------------------------------------
// Inlined function always reaches `analyzer-max-loop`, which stops the
// analysis on that path and puts the function on the "do not inline" list.

int inner_fixed_loop_1(int callIdx) {
  int i;
  clang_analyzer_dump(callIdx); // expected-warning {{1 S32}}
  for (i = 0; i < 10; i++); // FIXME-BUT-NEEDED: This stops the analysis.
  clang_analyzer_dump(callIdx); // no-warning
  return 42;
}

int outer_fixed_loop_1(void) {
  int x = inner_fixed_loop_1(1);
  int y = inner_fixed_loop_1(2);

 // FIXME-BUT-NEEDED: The analysis doesn't reach this zero division.
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function always reaches `analyzer-max-loop`; inlining is prevented
// even for different entry points.
// NOTE: the analyzer happens to analyze the entry points in a reversed order,
// so `outer_2_fixed_loop_2` is analyzed first and it will be the one which is
// able to inline the inner function.

int inner_fixed_loop_2(int callIdx) {
  // Identical copy of inner_fixed_loop_1.
  int i;
  clang_analyzer_dump(callIdx); // expected-warning {{2 S32}}
  for (i = 0; i < 10; i++); // FIXME-BUT-NEEDED: This stops the analysis.
  clang_analyzer_dump(callIdx); // no-warning
  return 42;
}

int outer_1_fixed_loop_2(void) {
  return inner_fixed_loop_2(1);
}

int outer_2_fixed_loop_2(void) {
  return inner_fixed_loop_2(2);
}

//-----------------------------------------------------------------------------
// Inlined function reaches `analyzer-max-loop` only in its second call. The
// function is inlined twice but the second call doesn't finish and ends up
// being conservatively evaluated.

int inner_parametrized_loop_1(int count) {
  int i;
  clang_analyzer_dump(count); // expected-warning {{2 S32}}
                              // expected-warning@-1 {{10 S32}}
  for (i = 0; i < count; i++);
      // FIXME-BUT-NEEDED: This loop stops the analysis when count >=4.
  clang_analyzer_dump(count); // expected-warning {{2 S32}}
  return 42;
}

int outer_parametrized_loop_1(void) {
  int x = inner_parametrized_loop_1(2);
  int y = inner_parametrized_loop_1(10);

 // FIXME-BUT-NEEDED: The analysis doesn't reach this zero division.
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function reaches `analyzer-max-loop` on its first call, so the
// second call isn't inlined (although it could be fully evaluated).

int inner_parametrized_loop_2(int count) {
  // Identical copy of inner_parametrized_loop_1.
  int i;
  clang_analyzer_dump(count); // expected-warning {{10 S32}}
  for (i = 0; i < count; i++);
      // FIXME-BUT-NEEDED: This loop stops the analysis when count >=4.
  clang_analyzer_dump(count); // no-warning
  return 42;
}

int outer_parametrized_loop_2(void) {
  int y = inner_parametrized_loop_2(10);
  int x = inner_parametrized_loop_2(2);

 // FIXME-BUT-NEEDED: The analysis doesn't reach this zero division.
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function may or may not reach `analyzer-max-loop` depending on an
// ambiguous check before the loop. This is very similar to the "fixed loop"
// cases: the function is placed on the "don't inline" list when any execution
// path reaches `analyzer-max-loop` (even if other execution paths reach the
// end of the function).
// NOTE: This is tested with two separate entry points to ensure that one
// inlined call is fully evaluated before we try to inline the other call.
// NOTE: the analyzer happens to analyze the entry points in a reversed order,
// so `outer_2_conditional_loop` is analyzed first and it will be the one which
// is able to inline the inner function.

int inner_conditional_loop(int callIdx) {
  int i;
  clang_analyzer_dump(callIdx); // expected-warning {{2 S32}}
  if (getNum() == 777) {
    for (i = 0; i < 10; i++);
  }
  clang_analyzer_dump(callIdx); // expected-warning {{2 S32}}
  return 42;
}

int outer_1_conditional_loop(void) {
  return inner_conditional_loop(1);
}

int outer_2_conditional_loop(void) {
  return inner_conditional_loop(2);
}

//-----------------------------------------------------------------------------
// Inlined function executes an ambiguous loop that may or may not reach
// `analyzer-max-loop`. Historically, before the "don't assume third iteration"
// commit (bb27d5e5c6b194a1440b8ac4e5ace68d0ee2a849) this worked like the
// `conditional_loop` cases: the analyzer was able to find a path reaching
// `analyzer-max-loop` so inlining was disabled. After that commit the analyzer
// does not _assume_ a third (or later) iteration (i.e. does not enter those
// iterations if the loop condition is an unknown value), so e.g. this test
// function does not reach `analyzer-max-loop` iterations and the inlining is
// not disabled.
// Unfortunately this change significantly increased the workload and
// runtime of the analyzer (more entry points used up their budget), so the
// option `inline-functions-with-ambiguous-loops` was introduced and disabled
// by default to suppress the inlining in situations where the "don't assume
// third iteration" logic activates.
// NOTE: This is tested with two separate entry points to ensure that one
// inlined call is fully evaluated before we try to inline the other call.
// NOTE: the analyzer happens to analyze the entry points in a reversed order,
// so `outer_2_ambiguous_loop` is analyzed first and it will be the one which
// is able to inline the inner function.

int inner_ambiguous_loop(int callIdx) {
  int i;
  clang_analyzer_dump(callIdx); // default-warning {{2 S32}}
                                // enabled-warning@-1 {{1 S32}}
                                // enabled-warning@-2 {{2 S32}}
  for (i = 0; i < getNum(); i++);
  return i;
}

int outer_1_ambiguous_loop(void) {
  return inner_ambiguous_loop(1);
}
int outer_2_ambiguous_loop(void) {
  return inner_ambiguous_loop(2);
}
