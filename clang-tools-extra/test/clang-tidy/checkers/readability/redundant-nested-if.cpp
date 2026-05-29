// RUN: %check_clang_tidy -std=c++98-or-later %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing

bool cond(int X = 0);
int side_effect();
void sink();
void bar();

struct BoolLike {
  operator bool() const;
};

BoolLike make_bool_like();

#define INNER_IF(C) if (C) sink()
#define COND_MACRO cond()
#define OUTER_IF if (cond())

void positive_cases() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) {
      sink();
    }
  }
  // CHECK-FIXES: if ((cond()) && (cond(1)))
  // CHECK-FIXES: sink();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) {
      // CHECK-MESSAGES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES: if ((cond()) && (cond(1)) && (cond(2)))
  // CHECK-FIXES: sink();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond() || cond(1))
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(2))
      sink();
  // CHECK-FIXES: if ((cond() || cond(1)) && (cond(2)))
  // CHECK-FIXES: sink();
}

void stress_long_chain_case() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond(0)) {
    if (cond(1)) {
      if (cond(2)) {
        if (cond(3)) {
          if (cond(4)) {
            if (cond(5)) {
              if (cond(6)) {
                if (cond(7)) {
                  if (cond(8)) {
                    if (cond(9)) {
                      if (cond(10)) {
                        if (cond(11))
                          sink();
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // CHECK-FIXES: if ((cond(0)) && (cond(1)) && (cond(2)) && (cond(3)) && (cond(4)) && (cond(5)) && (cond(6)) && (cond(7)) && (cond(8)) && (cond(9)) && (cond(10)) && (cond(11)))
  // CHECK-FIXES: sink();
}

void nested_chains_are_diagnosed_once_per_chain() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) {
      sink();
      // CHECK-MESSAGES: :[[@LINE+1]]:7: warning: nested 'if' statements can be merged together
      if (cond(2)) {
        // CHECK-MESSAGES: :[[@LINE+1]]:9: note: nested 'if' statement to merge declared here
        if (cond(3))
          sink();
      }
    }
  }
  // CHECK-FIXES: if ((cond()) && (cond(1)))
  // CHECK-FIXES: if ((cond(2)) && (cond(3)))
}

void child_chain_is_reported_when_parent_is_not_diagnosable() {
  // CHECK-MESSAGES: :[[@LINE+2]]:5: warning: nested 'if' statements can be merged together
  OUTER_IF {
    if (cond(1)) {
      // CHECK-MESSAGES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES: OUTER_IF {
  // CHECK-FIXES: if ((cond(1)) && (cond(2)))
  // CHECK-FIXES: sink();
}

void else_branch_child_chain_is_reported_when_parent_is_not_diagnosable() {
  // CHECK-MESSAGES: :[[@LINE+4]]:5: warning: nested 'if' statements can be merged together
  if (cond()) {
    sink();
  } else {
    if (cond(1)) {
      // CHECK-MESSAGES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES: } else {
  // CHECK-FIXES: if ((cond(1)) && (cond(2)))
  // CHECK-FIXES: sink();
}

void negative_cases() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    if (bool B = cond(1))
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    if (cond(1))
      sink();
    else
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    if (cond(1))
      sink();
  } else {
    sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    sink();
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    if (cond(1))
      sink();
    sink();
  }
}

void macro_and_preprocessor_cases() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    INNER_IF(cond(1));
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (COND_MACRO) {
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
#if 1
    if (cond(1))
      sink();
#endif
  }
}

void comment_handling_cases() {
  // Comments inside condition payloads are preserved by the merged condition.
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond() /* outer payload */) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (/* inner payload */ cond(1))
      sink();
  }
  // CHECK-FIXES: if ((cond() /* outer payload */) && (/* inner payload */ cond(1)))
  // CHECK-FIXES: sink();

  // Trailing comments in nested headers keep the diagnostic but suppress fix-its.
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) // outer trailing
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) // inner trailing
      sink();
  // CHECK-FIXES: if (cond()) // outer trailing
  // CHECK-FIXES: if (cond(1)) // inner trailing
  // CHECK-FIXES: sink();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if /* nested header comment */ (cond(1))
      sink();
  }
  // CHECK-FIXES: if /* nested header comment */ (cond(1))
}

void user_defined_bool_conversion_default_cases() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (make_bool_like()) {
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond(1)) {
    if (make_bool_like())
      sink();
  }
}
