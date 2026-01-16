// RUN: %check_clang_tidy -std=c99,c11,c17 %s readability-implicit-bool-conversion %t

#define true 1
#define false 0

_Bool returns_bool(void) { return true; }
int returns_int(void) { return 1; }

void test_c99_logical_ops(void) {
  _Bool b1 = true;
  _Bool b2 = false;

  if (b1 && b2) {}
  if (b1 && returns_int()) {}
}

void test_c99_comparison(void) {
  int x = 1;
  int y = 2;
  _Bool b = x > y;
}

void test_c99_native_keyword(void) {
  _Bool raw_bool = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'int' -> 'bool' [readability-implicit-bool-conversion]
  // CHECK-FIXES: _Bool raw_bool = true;
  int i = raw_bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int i = (int)raw_bool;
}

void test_c99_macro_behavior(void) {
  _Bool b = true;
  int i = b + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int i = (int)b + 1;
}

void test_c99_pointer_conversion(int *p) {
  _Bool b = p;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: _Bool b = p != 0;
}
