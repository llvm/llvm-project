// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-max-loop 4 -analyzer-config widen-loops=true -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -DTEST_NULL_TERM -analyzer-checker=core,unix.Malloc,debug.ExprInspection,alpha.cplusplus.IteratorRange -analyzer-max-loop 4 -analyzer-config widen-loops=true -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

void loop_which_iterates_limit_times_not_widened() {
  int i;
  int x = 1;
  // Check loop isn't widened by checking x isn't invalidated
  for (i = 0; i < 1; ++i) {}
  clang_analyzer_eval(x == 1); // expected-warning {{TRUE}}
  for (i = 0; i < 2; ++i) {}
  clang_analyzer_eval(x == 1); // expected-warning {{TRUE}}
  for (i = 0; i < 3; ++i) {}
  // FIXME loss of precision as a result of evaluating the widened loop body
  //       *instead* of the last iteration.
  clang_analyzer_eval(x == 1); // expected-warning {{UNKNOWN}}
}

int a_global;

void loop_evaluated_before_widening() {
  int i;
  a_global = 1;
  for (i = 0; i < 10; ++i) {
    if (i == 2) {
      // True before widening then unknown after.
      clang_analyzer_eval(a_global == 1); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}}
    }
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void warnings_after_loop() {
  int i;
  for (i = 0; i < 10; ++i) {}
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void for_loop_exits() {
  int i;
  for (i = 0; i < 10; ++i) {}
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void while_loop_exits() {
  int i = 0;
  while (i < 10) {++i;}
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void do_while_loop_exits() {
  int i = 0;
  do {++i;} while (i < 10);
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void loop_body_is_widened() {
  int i = 0;
  while (i < 100) {
    if (i > 10) {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
    ++i;
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void invariably_infinite_loop() {
  int i = 0;
  while (1) { ++i; }
  clang_analyzer_warnIfReached(); // no-warning
}

void invariably_infinite_break_loop() {
  int i = 0;
  while (1) {
    ++i;
    int x = 1;
    if (!x) break;
  }
  clang_analyzer_warnIfReached(); // no-warning
}

void reachable_break_loop() {
  int i = 0;
  while (1) {
    ++i;
    if (i == 100) break;
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void condition_constrained_true_in_loop() {
  int i = 0;
  while (i < 50) {
    clang_analyzer_eval(i < 50); // expected-warning {{TRUE}}
    ++i;
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void condition_constrained_false_after_loop() {
  int i = 0;
  while (i < 50) {
    ++i;
  }
  clang_analyzer_eval(i >= 50); // expected-warning {{TRUE}}
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void multiple_exit_test() {
  int x = 0;
  int i = 0;
  while (i < 50) {
    if (x) {
      i = 10;
      break;
    }
    ++i;
  }
  // Reachable by 'normal' exit
  if (i == 50) clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  // Reachable by break point
  if (i == 10) clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  // Not reachable
  if (i < 10) clang_analyzer_warnIfReached(); // no-warning
  if (i > 10 && i < 50) clang_analyzer_warnIfReached(); // no-warning
}

void pointer_doesnt_leak_from_loop() {
  int *h_ptr = (int *) malloc(sizeof(int));
  for (int i = 0; i < 2; ++i) {}
  for (int i = 0; i < 10; ++i) {} // no-warning
  free(h_ptr);
}

int g_global;

void unknown_after_loop(int s_arg) {
  g_global = 0;
  s_arg = 1;
  int s_local = 2;
  int *h_ptr = malloc(sizeof(int));

  for (int i = 0; i < 10; ++i) {}

  clang_analyzer_eval(g_global); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(s_arg); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(s_local); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(h_ptr == 0); // expected-warning {{UNKNOWN}}
  free(h_ptr);
}

void variable_bound_exiting_loops_widened(int x) {
  int i = 0;
  int t = 1;
  while (i < x) {
    ++i;
  }
  clang_analyzer_eval(t == 1); // expected-warning {{TRUE}} // expected-warning {{UNKNOWN}}
}

void nested_loop_outer_widen() {
  int i = 0, j = 0;
  for (i = 0; i < 10; i++) {
    clang_analyzer_eval(i < 10); // expected-warning {{TRUE}}
    for (j = 0; j < 2; j++) {
      clang_analyzer_eval(j < 2); // expected-warning {{TRUE}}
    }
    clang_analyzer_eval(j >= 2); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(i >= 10); // expected-warning {{TRUE}}
}

void nested_loop_inner_widen() {
  int i = 0, j = 0;
  for (i = 0; i < 2; i++) {
    clang_analyzer_eval(i < 2); // expected-warning {{TRUE}}
    for (j = 0; j < 10; j++) {
      clang_analyzer_eval(j < 10); // expected-warning {{TRUE}}
    }
    clang_analyzer_eval(j >= 10); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(i >= 2); // expected-warning {{TRUE}}
}

#ifdef TEST_NULL_TERM
void null_terminator_loop_widen(int *a) {
  int c;
  // Loop widening will call 'invalidateRegions()' and 'invalidateRegions()'
  // will construct the SymbolConjured with null Stmt because of the null
  // terminator statement. Accessing the null Stmt will cause a crash.
  for (;;) {
    c = *a; // no-crash
    a++;
  }
}
#endif
