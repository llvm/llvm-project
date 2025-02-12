// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection \
// RUN:     -verify=expected,eagerlyassume %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection \
// RUN:     -analyzer-config eagerly-assume=false \
// RUN:     -verify=expected,noeagerlyassume %s

// These tests validate the logic within `ExprEngine::processBranch` which
// ensures that in loops with opaque conditions we don't assume execution paths
// if the code does not imply that they are possible.

void clang_analyzer_numTimesReached(void);
void clang_analyzer_warnIfReached(void);
void clang_analyzer_dump(int);

void clearCondition(void) {
  // If the analyzer can definitely determine the value of the loop condition,
  // then this corrective logic doesn't activate and the engine executes
  // `-analyzer-max-loop` iterations (by default, 4).
  for (int i = 0; i < 10; i++)
    clang_analyzer_numTimesReached(); // expected-warning {{4}}

  clang_analyzer_warnIfReached(); // unreachable
}

void opaqueCondition(int arg) {
  // If the loop condition is opaque, don't assume more than two iterations,
  // because the presence of a loop does not imply that the programmer thought
  // that more than two iterations are possible. (It _does_ imply that two
  // iterations may be possible at least in some cases, because otherwise an
  // `if` would've been enough.)
  for (int i = 0; i < arg; i++)
    clang_analyzer_numTimesReached(); // expected-warning {{2}}

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

int check(void);

void opaqueConditionCall(int arg) {
  // Same situation as `opaqueCondition()` but with a `while ()` loop. This
  // is also an example for a situation where the programmer cannot easily
  // insert an assertion to guide the analyzer and rule out more than two
  // iterations (so the analyzer needs to proactively avoid those unjustified
  // branches).
  while (check())
    clang_analyzer_numTimesReached(); // expected-warning {{2}}

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void opaqueConditionDoWhile(int arg) {
  // Same situation as `opaqueCondition()` but with a `do {} while ()` loop.
  // This is tested separately because this loop type is a special case in the
  // iteration count calculation.
  int i = 0;
  do {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
  } while (i++ < arg);

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void dontRememberOldBifurcation(int arg) {
  // In this (slightly contrived) test case the analyzer performs an assumption
  // at the first iteration of the loop, but does not make any new assumptions
  // in the subsequent iterations, so the analyzer should continue evaluating
  // the loop.
  // Previously this was mishandled in `eagerly-assume` mode (which is enabled
  // by default), because the code remembered that there was a bifurcation on
  // the first iteration of the loop and didn't realize that this is obsolete.

  // NOTE: The variable `i` is introduced to ensure that the iterations of the
  // loop change the state -- otherwise the analyzer stops iterating because it
  // returns to the same `ExplodedNode`.
  int i = 0;
  while (arg > 3) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    i++;
  }

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void dontAssumeFourthIterartion(int arg) {
  if (arg == 2)
    return;

  // In this function the analyzer cannot leave the loop after exactly two
  // iterations (because it knows that `arg != 2` at that point), so it
  // performs a third iteration, but it does not assume that a fourth iteration
  // is also possible.
  for (int i = 0; i < arg; i++)
    clang_analyzer_numTimesReached(); // expected-warning {{3}}

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

#define TRUE 1
void shortCircuitInLoopCondition(int arg) {
  // When the loop condition expression contains short-circuiting operators, it
  // performs "inner" bifurcations for those operators and only considers the
  // last (rightmost) operand as the branch condition that is associated with
  // the loop itself (as its loop condition).
  // This means that assumptions taken in the left-hand side of a short-circuiting
  // operator are not recognized as "opaque" loop condition, so the loop in
  // this test case is allowed to finish four iterations.
  // FIXME: This corner case is responsible for at least one out-of-bounds
  // false positive on the ffmpeg codebase. Eventually we should properly
  // recognize the full syntactical loop condition expression as "the loop
  // condition", but this will be complicated to implement.
  for (int i = 0; i < arg && TRUE; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void shortCircuitInLoopConditionRHS(int arg) {
  // Unlike `shortCircuitInLoopCondition()`, this case is handled properly
  // because the analyzer thinks that the right hand side of the `&&` is the
  // loop condition.
  for (int i = 0; TRUE && i < arg; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void eagerlyAssumeInSubexpression(int arg) {
  // The `EagerlyAssume` logic is another complication that can "split the
  // state" within the loop condition, but before the `processBranch()` call
  // which is (in theory) responsible for evaluating the loop condition.
  // The current implementation partially compensates this by noticing the
  // cases where the loop condition is targeted by `EagerlyAssume`, but does
  // not handle the (fortunately rare) case when `EagerlyAssume` hits a
  // sub-expression of the loop condition (as in this contrived test case).
  // FIXME: I don't know a real-world example for this inconsistency, but it
  // would be good to eliminate it eventually.
  for (int i = 0; (i >= arg) - 1; i++) {
    clang_analyzer_numTimesReached(); // eagerlyassume-warning {{4}} noeagerlyassume-warning {{2}}
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void calledTwice(int arg, int isFirstCall) {
  // This function is called twice (with two different unknown 'arg' values) to
  // check the iteration count handling in this situation.
  for (int i = 0; i < arg; i++) {
    if (isFirstCall) {
      clang_analyzer_numTimesReached(); // expected-warning {{2}}
    } else {
      clang_analyzer_numTimesReached(); // expected-warning {{2}}
    }
  }
}

void caller(int arg, int arg2) {
  // Entry point for `calledTwice()`.
  calledTwice(arg, 1);
  calledTwice(arg2, 0);
}

void innerLoopClearCondition(void) {
  // A "control group" test case for the behavior of an inner loop. Notice that
  // although the (default) value of `-analyzer-max-loop` is 4, we only see 3 iterations
  // of the inner loop, because `-analyzer-max-loop` limits the number of
  // evaluations of _the loop condition of the inner loop_ and in addition to
  // the 3 evaluations before the 3 iterations, there is also a step where it
  // evaluates to false (in the first iteration of the outer loop).
  for (int outer = 0; outer < 2; outer++) {
    int limit = 0;
    if (outer)
      limit = 10;
    clang_analyzer_dump(limit); // expected-warning {{0}} expected-warning {{10}}
    for (int i = 0; i < limit; i++) {
      clang_analyzer_numTimesReached(); // expected-warning {{3}}
    }
  }
}

void innerLoopOpaqueCondition(int arg) {
  // In this test case the engine doesn't assume a second iteration within the
  // inner loop (in the second iteration of the outer loop, when the limit is
  // opaque) because `CoreEngine::getCompletedIterationCount()` is based on the
  // `BlockCount` values queried from the `BlockCounter` which count _all_
  // evaluations of a given `CFGBlock` (in our case, the loop condition) and
  // not just the evaluations within the current iteration of the outer loop.
  // FIXME: This inaccurate iteration count could in theory cause some false
  // negatives, although I think this would be unusual in practice, as the
  // small default value of `-analyzer-max-loop` means that this is only
  // relevant if the analyzer can deduce that the inner loop performs 0 or 1
  // iterations within the first iteration of the outer loop (and then the
  // condition of the inner loop is opaque within the second iteration of the
  // outer loop).
  for (int outer = 0; outer < 2; outer++) {
    int limit = 0;
    if (outer)
      limit = arg;
    clang_analyzer_dump(limit); // expected-warning {{0}} expected-warning {{reg_$}}
    for (int i = 0; i < limit; i++) {
      clang_analyzer_numTimesReached(); // expected-warning {{1}}
    }
  }
}

void onlyLoopConditions(int arg) {
  // This "don't assume third iteration" logic only examines the conditions of
  // loop statements and does not affect the analysis of code that implements
  // similar behavior with different language features like if + break, goto,
  // recursive functions, ...
  int i = 0;
  while (1) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}

    // This is not a loop condition.
    if (i++ > arg)
      break;
  }

  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}
