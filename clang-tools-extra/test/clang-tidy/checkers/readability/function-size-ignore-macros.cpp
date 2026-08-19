// RUN: %check_clang_tidy -check-suffix=IGNORE-MACROS %s readability-function-size %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-function-size.StatementThreshold: 1, \
// RUN:         readability-function-size.BranchThreshold: 0, \
// RUN:         readability-function-size.NestingThreshold: 1, \
// RUN:         readability-function-size.VariableThreshold: 0, \
// RUN:         readability-function-size.ParameterThreshold: 5, \
// RUN:         readability-function-size.IgnoreMacros: true \
// RUN:     }}'

#define STATEMENT ;
#define BRANCH if (true) {}
#define VARIABLE int X;
#define BODY ;
#define INIT_X X(0)
#define INIT_Y Y(0)
#define  WRAP(x) x

void user_statements() {
  ;
}

void user_branch() {
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-1]]:6: warning: function 'user_branch' exceeds recommended size/complexity thresholds [readability-function-size]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: note: 2 statements (threshold 1)
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-3]]:6: note: 1 branches (threshold 0)
  if (true) {
    // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-1]]:13: note: nesting level 2 starts here (threshold 1)
  }
}

void user_variable() {
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-1]]:6: warning: function 'user_variable' exceeds recommended size/complexity thresholds [readability-function-size]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: note: 1 variables (threshold 0)
  int X;
}

void macro_statement() {
  ;
  STATEMENT
}

void macro_branch() {
  ;
  BRANCH
}

void macro_variable() {
  ;
  VARIABLE
}

void macro_body_too_many_params(int a, int b, int c, int d, int e, int f) {
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-1]]:6: warning: function 'macro_body_too_many_params' exceeds recommended size/complexity thresholds [readability-function-size]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: note: 6 parameters (threshold 5)
  ;
  BODY
}

struct MacroCtorInit {
  int X, Y;
  MacroCtorInit() : INIT_X, INIT_Y {
    ;
  }
};

void macro_wrap() {
  // The macro argument written by the user is currently ignored.
  ;
  WRAP(if (true){})
}
