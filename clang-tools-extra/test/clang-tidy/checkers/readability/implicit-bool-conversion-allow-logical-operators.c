// RUN: %check_clang_tidy -std=c23-or-later %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.AllowLogicalOperatorConversion: true \
// RUN:     }}'

void function_taking_bool(bool);
bool returns_bool(void);
int returns_int(void);

void logical_or_to_bool(void) {
  bool a = true, b = false;
  bool c = a || b;
  bool d = returns_bool() || returns_bool();
  bool e = returns_bool() || (a && b);
}

void logical_and_to_bool(void) {
  bool a = true, b = false;
  bool c = a && b;
  bool d = returns_bool() && returns_bool();
}

void logical_not_to_bool(void) {
  bool a = true;
  int x = 5;
  bool b = !a;
  bool c = !x;
}

void logical_with_literals(void) {
  bool a = true, b = false;
  bool c = true || b;
  bool d = false || b;
  bool e = a || true;
  bool f = false || (a || b);
}

void nested_logical_ops(void) {
  bool a = true, b = false, c = true;
  bool d = (a && b) || c;
  bool e = a || (b && c);
  bool f = !(a || b);
}

void logical_in_function_call(void) {
  bool a = true, b = false;
  function_taking_bool(a || b);
  function_taking_bool(a && b);
  function_taking_bool(!a);
}

bool logical_in_return(void) {
  bool a = true, b = false;
  return a || b;
}

void still_warn_on_regular_int_to_bool(void) {
  int x = 42;
  bool b = x;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool' [readability-implicit-bool-conversion]
  // CHECK-FIXES: bool b = x != 0;

  bool c = x + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool c = (x + 1) != 0;

  bool d = returns_int();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool d = returns_int() != 0;
}

void still_warn_on_bitwise_ops(void) {
  int x = 5, y = 3;
  bool b = x | y;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool b = (x | y) != 0;

  bool c = x & y;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool c = (x & y) != 0;
}

void cast_from_bool_still_warns(void) {
  bool a = true;
  int x = a;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int x = (int)a;
}

void comparison_still_excluded(void) {
  bool b1 = 1 > 0;
  bool b2 = 1 == 0;
  bool b3 = 1 < 2;
}
