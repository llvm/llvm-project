// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify=expected,default %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config legacy-inlining-prevention=false -verify=expected,disabled %s

int getNum(void); // Get an opaque number.

void clang_analyzer_numTimesReached(void);
void clang_analyzer_dump(int arg);

//-----------------------------------------------------------------------------
// Simple case: inlined function never reaches `analyzer-max-loop`.

int inner_simple(void) {
  clang_analyzer_numTimesReached(); // expected-warning {{2}}
  return 42;
}

int outer_simple(void) {
  int x = inner_simple();
  int y = inner_simple();
  return 53 / (x - y); // expected-warning {{Division by zero}}
}

//-----------------------------------------------------------------------------
// Inlined function always reaches `analyzer-max-loop`.

int inner_fixed_loop_1(void) {
  int i;
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  for (i = 0; i < 10; i++);
  clang_analyzer_numTimesReached(); // no-warning
  return 42;
}

int outer_fixed_loop_1(void) {
  int x = inner_fixed_loop_1();
  int y = inner_fixed_loop_1();
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function always reaches `analyzer-max-loop`; inlining is prevented
// even for different entry points.
// This test uses `clang_analyzer_dump` and distinct `arg` values because
// `clang_analyzer_numTimesReached` only counts the paths reaching that node
// during the analysis of one particular entry point, so it cannot distinguish
// "two entry points reached this, both with one path" (where the two reports
// are unified as duplicates by the generic report postprocessing) and "one
// entry point reached this with one path" (where naturally nothing shows that
// the second entry point _tried_ to reach it).

int inner_fixed_loop_2(int arg) {
  // Identical copy of inner_fixed_loop_1
  int i;
  clang_analyzer_dump(arg); // expected-warning {{2}}
  for (i = 0; i < 10; i++);
  clang_analyzer_dump(arg); // no-warning
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
  clang_analyzer_numTimesReached(); // expected-warning {{2}}
  for (i = 0; i < count; i++);
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  return 42;
}

int outer_parametrized_loop_1(void) {
  int x = inner_parametrized_loop_1(2);
  int y = inner_parametrized_loop_1(10);
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function reaches `analyzer-max-loop` on its first call, so the
// second call isn't inlined (although it could be fully evaluated).

int inner_parametrized_loop_2(int count) {
  int i;
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  for (i = 0; i < count; i++);
  clang_analyzer_numTimesReached(); // no-warning
  return 42;
}

int outer_parametrized_loop_2(void) {
  int y = inner_parametrized_loop_2(10);
  int x = inner_parametrized_loop_2(2);
  return 53 / (x - y); // no-warning
}

//-----------------------------------------------------------------------------
// Inlined function may or may not reach `analyzer-max-loop` depending on an
// opaque check before the loop. This is very similar to the "fixed loop"
// cases: the function is placed on the "don't inline" list when any execution
// path reaches `analyzer-max-loop` (even if other execution paths reach the
// end of the function).

int inner_conditional_loop(void) {
  int i;
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  if (getNum() == 777) {
    for (i = 0; i < 10; i++);
  }
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  return 42;
}

int outer_1_conditional_loop(void) {
  return inner_conditional_loop();
}

int outer_2_conditional_loop(void) {
  return inner_conditional_loop();
}

//-----------------------------------------------------------------------------
// Inlined function executes an opaque loop that may or may not reach
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
// option `legacy-inlining-prevention` was introduced and enabled by default to
// suppress the inlining in situations where the "don't assume third iteration"
// logic activates.
// This testcase demonstrate that the inlining is prevented with the default
// `legacy-inlining-prevention=true` config, but is not prevented when this
// option is disabled (set to false).

int inner_opaque_loop_1(void) {
  int i;
  clang_analyzer_numTimesReached(); // default-warning {{1}} disabled-warning {{2}}
  for (i = 0; i < getNum(); i++);
  return i;
}

int outer_opaque_loop_1(void) {
  int iterCount = inner_opaque_loop_1();

  // The first call to `inner_opaque_loop_1()` splits three execution paths that
  // differ in the number of performed iterations (0, 1 or 2). The function
  // `inner_opaque_loop_1` is added to the "do not inline this" list when the
  // path that performed two iterations tries to enter the third iteration (and
  // the "don't assume third iteration" logic prevents this) -- but the other
  // two paths (which performed 0 and 1 iterations) would reach and inline the
  // second `inner_opaque_loop_1()` before this would happen (because the
  // default traversal is a complex heuristic that happens to prefer this). The
  // following `if` will discard these "early exit" paths to highlight the
  // difference between the default and disabled state:
  if (iterCount < 2)
    return 0;

  return inner_opaque_loop_1();
}

//-----------------------------------------------------------------------------
// Another less contrived testcase that demonstrates the difference between the
// enabled (default) and disabled state of `legacy-inlining-prevention`.
// Here the two calls to `inner_opaque_loop_2()` are in different entry points
// so the first call is fully analyzed (and can put the function on the "do
// not inline" list) before reaching the second call.
// This test uses `clang_analyzer_dump` because (as explained in an earlier
// comment block) `clang_analyzer_numTimesReached` is not suitable for counting
// visits from separate entry points.

int inner_opaque_loop_2(int arg) {
  int i;
  clang_analyzer_dump(arg); // default-warning {{2}}
                            // disabled-warning@-1 {{1}} disabled-warning@-1 {{2}}
  for (i = 0; i < getNum(); i++);
  return i;
}

int outer_1_opaque_loop_2(void) {
  return inner_opaque_loop_2(1);
}
int outer_2_opaque_loop(void) {
  return inner_opaque_loop_2(2);
}
