// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config unroll-loops=true,cfg-loopexit=true -verify -std=c++14 -analyzer-config exploration_strategy=unexplored_first_queue %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config unroll-loops=true,cfg-loopexit=true,exploration_strategy=dfs -verify -std=c++14 -DDFS=1 %s

void clang_analyzer_numTimesReached();
void clang_analyzer_warnIfReached();

int getNum();
void foo(int &);

int simple_unroll1() {
  int a[9];
  int k = 42;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_unroll2() {
  int a[9];
  int k = 42;
  int i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    a[i] = 42;
  }

  for (int j = 0; j <= 9; ++j) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
    a[j] = 42;
  }

  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_unroll3_unsigned() {
  int a[9];
  int k = 42;
  for (unsigned i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_unroll4_unsigned() {
  int a[9];
  int k = 42;
  unsigned i;
  for (i = (0); i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_no_unroll1() {
  int a[9];
  int k = 42;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
    a[i] = 42;
    foo(i);
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_no_unroll2() {
  int a[9];
  int k = 42;
  int i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
    a[i] = 42;
    i += getNum();
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_no_unroll3() {
  int a[9];
  int k = 42;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
    (void)&i;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int simple_no_unroll4() {
  int a[9];
  int k = 42;
  int i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
    int &j = i;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int simple_no_unroll5() {
  int a[9];
  int k = 42;
  int i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
    int &j{i};
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int no_unroll_assignment() {
  for (int i = 0; i < 9; i++) {
    i = i + 1;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment2() {
  for (int i = 0; i < 9; i++) {
    i *= 2;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment3() {
  for (int i = 128; i > 0; i--) {
    i /= 2;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment4() {
  for (int i = 0; i < 9; i++) {
    i -= 2;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment5() {
  for (int i = 0; i < 9; i++) {
    i += 1;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment6() {
  for (int i = 128; i > 0; i--) {
    i >>= 1;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment7() {
  for (int i = 0; i < 512; i++) {
    i <<= 1;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment8() {
  for (int i = 0; i < 9; i++) {
    i %= 8;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment9() {
  for (int i = 0; i < 9; i++) {
    i &= 31;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment10() {
  for (int i = 0; i < 9; i++) {
    i |= 2;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int no_unroll_assignment11() {
  for (int i = 0; i < 9; i++) {
    i ^= 2;
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int make_new_branches_loop_cached() {
  for (int i = 0; i < 8; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    if (getNum()) {
      (void)i; // Since this Stmt does not change the State the analyzer
               // won't make a new execution path but reuse the earlier nodes.
    }
  }
  clang_analyzer_warnIfReached(); // no-warning
  return 0;
}

int make_new_branches_loop_uncached() {
  int l = 2;
  for (int i = 0; i < 8; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
    if (getNum()) {
      ++l;
    }
  }
  clang_analyzer_warnIfReached(); // no-warning
  return 0;
}

int make_new_branches_loop_uncached2() {
  int l = 2;
  for (int i = 0; i < 8; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
    if (getNum()) {
      ++l;
    }
    (void)&i; // This ensures that the loop won't be unrolled.
  }
  clang_analyzer_warnIfReached(); // no-warning
  return 0;
}


int escape_before_loop_no_unroll1() {
  int a[9];
  int k = 42;
  int i;
  int &j = i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int escape_before_loop_no_unroll2() {
  int a[9];
  int k = 42;
  int i;
  int *p = &i;
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int escape_before_loop_no_unroll3() {
  int a[9];
  int k = 42;
  int i;
  foo(i);
  for (i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    a[i] = 42;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int nested_outer_unrolled() {
  int a[9];
  int k = 42;
  int j = 0;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{1}}
    for (j = 0; j < 9; ++j) {
      clang_analyzer_numTimesReached(); // expected-warning {{4}}
      a[j] = 22;
      (void)&j; // ensures that the inner loop won't be unrolled
    }
    a[i] = 42;
  }
  int b = 22 / (k - 42); // no-warning
  return 0;
}

int nested_inner_unrolled() {
  int a[9];
  int k = 42;
  int j = 0;
  for (int i = 0; i < getNum(); i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
    for (j = 0; j < 8; ++j) {
      clang_analyzer_numTimesReached(); // expected-warning {{16}}
      a[j] = 22;
    }
    a[i] = 42;
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int nested_both_unrolled() {
  int a[9];
  int k = 42;
  int j = 0;
  for (int i = 0; i < 7; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{7}}
    for (j = 0; j < 6; ++j) {
      clang_analyzer_numTimesReached(); // expected-warning {{42}}
      a[j] = 22;
    }
    a[i] = 42;
  }
  int b = 22 / (k - 42); // expected-warning {{Division by zero}}
  return 0;
}

int simple_known_bound_loop() {
  for (int i = 2; i < 12; i++) {
    // This function is inlined in nested_inlined_unroll1()
    clang_analyzer_numTimesReached(); // expected-warning {{90}}
  }
  return 0;
}

int simple_unknown_bound_loop() {
  for (int i = 2; i < getNum(); i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{8}}
  }
  return 0;
}

int nested_inlined_unroll1() {
  int k;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    k = simple_known_bound_loop();    // no reevaluation without inlining
  }
  int a = 22 / k; // expected-warning {{Division by zero}}
  return 0;
}

int nested_inlined_no_unroll1() {
  int k;
  for (int i = 0; i < 9; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
    k = simple_unknown_bound_loop();  // reevaluation without inlining, splits the state as well
  }
  int a = 22 / k; // no-warning
  return 0;
}

int recursion_unroll1(bool b) {
  int k = 2;
  for (int i = 0; i < 5; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{13}}
    if (i == 0 && b)                  // Splits the state in the first iteration but the recursion
                                      // call will be unrolled anyway since the condition is known there.
      recursion_unroll1(false);
    clang_analyzer_numTimesReached(); // expected-warning {{14}}
  }
  int a = 22 / k; // no-warning
  return 0;
}

int recursion_unroll2(bool b) {
  int k = 0;
  for (int i = 0; i < 5; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
    if (i == 0 && b)
      recursion_unroll2(false);
    clang_analyzer_numTimesReached(); // expected-warning {{9}}
  }
  int a = 22 / k; // expected-warning {{Division by zero}}
  return 0;
}

int recursion_unroll3(bool b) {
  int k = 2;
  for (int i = 0; i < 5; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
    if (i == 4 && b) {
      recursion_unroll3(false);
      break;
    }
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
  }
  int a = 22 / k;
  return 0;
}

int recursion_unroll4(bool b) {
  int k = 2;
  for (int i = 0; i < 5; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{13}}
    if (i == 0 && b) {
      recursion_unroll4(false);
      continue;
    }
    clang_analyzer_numTimesReached(); // expected-warning {{13}}
  }
  int a = 22 / k;
  return 0;
}

int loop_exit_while_empty_loop_stack() {
  if (getNum())
    for (int i = 1; i < 8; i++)
      ;
  return 0;
}

int num_steps_on_limit() {
  for (int i = 0; i < 128; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{128}}
  }
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  return 0;
}

int num_steps_over_limit1() {
  for (int i = 0; i < 129; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
  return 0;
}

int num_steps_on_limit2() {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 64; j++) {
      clang_analyzer_numTimesReached(); // expected-warning {{128}}
    }
  }
  return 0;
}

int num_steps_over_limit2() {
  for (int i = 0; i < 2; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{1}}
    for (int j = 0; j <= 64; j++) {
      clang_analyzer_numTimesReached(); // expected-warning {{4}}
    }
  }
  return 0;
}

int num_steps_on_limit3() {
  for (int i = 0; i < getNum(); i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{2}}
    for (int j = 0; j < 32; j++) {
      // Here the loop unrollig logic calculates with four potential iterations
      // in the outer loop where it cannot determine the iteration count in
      // advance; but after two loops the analyzer conservatively assumes that
      // the (still opaque) loop condition is false.
      clang_analyzer_numTimesReached(); // expected-warning {{64}}
    }
  }
  return 0;
}

int num_steps_over_limit3() {
  for (int i = 0; i < getNum(); i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{1}}
    for (int j = 0; j < 33; j++) {
      clang_analyzer_numTimesReached(); // expected-warning {{4}}
    }
  }
  return 0;
}

int num_steps_on_limit4() {
  for (int i = 0; i < 4; i++) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
    for (int j = 0; j < 32; j++) {
      clang_analyzer_numTimesReached(); // expected-warning {{128}}
    }
  }
  return 0;
}

void pr34943() {
  for (int i = 0; i < 6L; ++i) {
    clang_analyzer_numTimesReached(); // expected-warning {{6}}
  }
}

void parm_by_value_as_loop_counter(int i) {
  for (i = 0; i < 10; ++i) {
    clang_analyzer_numTimesReached(); // expected-warning {{10}}
  }
}

void parm_by_ref_as_loop_counter(int &i) {
  for (i = 0; i < 10; ++i) {
    clang_analyzer_numTimesReached(); // expected-warning {{4}}
  }
}

void capture_by_value_as_loop_counter() {
  int out = 0;
  auto l = [i = out]() mutable {
    for (i = 0; i < 10; ++i) {
      clang_analyzer_numTimesReached(); // expected-warning {{10}}
    }
  };
}

void capture_by_ref_as_loop_counter() {
  int out = 0;
  auto l = [&i = out]() {
    for (i = 0; i < 10; ++i) {
      clang_analyzer_numTimesReached(); // expected-warning {{4}}
    }
  };
}

void capture_implicitly_by_value_as_loop_counter() {
  int i = 0;
  auto l = [=]() mutable {
    for (i = 0; i < 10; ++i) {
      clang_analyzer_numTimesReached(); // expected-warning {{10}}
    }
  };
}

void capture_implicitly_by_ref_as_loop_counter() {
  int i = 0;
  auto l = [&]() mutable {
    for (i = 0; i < 10; ++i) {
      clang_analyzer_numTimesReached(); // expected-warning {{4}}
    }
  };
}


void test_escaping_on_var_before_switch_case_no_crash(int c) {
  // https://github.com/llvm/llvm-project/issues/68819
  switch (c) {
    int i; // no-crash: The declaration of `i` is found here.
    case 0: {
      for (i = 0; i < 16; i++) {}
      break;
    }
  }
}
