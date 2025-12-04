// RUN: %check_clang_tidy -std=c99 %s readability-implicit-bool-conversion %t

typedef _Bool bool;
#define true 1
#define false 0

bool returns_bool(void) { return true; }
int returns_int(void) { return 1; }

void test_c99_logical_ops(void) {
  bool b1 = true;
  bool b2 = false;

  if (b1 && b2) {}

  if (b1 && returns_int()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
  // CHECK-FIXES: if ((int)b1 && returns_int()) {}
}

void test_c99_comparison(void) {
  int x = 1;
  int y = 2;
  bool b = x > y;
}
