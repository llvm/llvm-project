// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t -- -- -std=c++17

void test_init_stmt_true() {
  void foo(int i);
  if (int i = 0; true)
    foo(i);
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
  // CHECK-FIXES:   { int i = 0; foo(i); };
}

void test_init_stmt_false() {
  void foo(int i);
  if (int i = 0; false)
    foo(i);
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
  // CHECK-FIXES:   { int i = 0; };
}

void if_with_init_statement() {
  bool x = true;
  if (bool y = x; y == true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES: if (bool y = x; y) {
  }
}

// This test verifies that we don't crash on C++17 init-statements with complex objects.
// We use function calls to prevent the "conditional assignment" check from triggering.
void test_cxx17_no_crash() {
  struct RAII {};
  bool Cond = true;
  void body();
  void else_body();

  if (RAII Object; Cond) {
    body();
  } else {
    else_body();
  }

  if (bool X = Cond; X) {
    body();
  } else {
    else_body();
  }
}
