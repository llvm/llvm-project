// RUN: %check_clang_tidy %s bugprone-assignment-in-selection-statement %t

void test_if(int a, int b, int c, int d) {
  if (a = b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: Assignment within condition of 'if' statement may indicate programming error
  if ((a = b)) {}
  if (a == b) {}

  if ((b > 0) ? (a = b) : c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: Assignment within condition of 'if' statement may indicate programming error
  if ((b > 0) ? c : (a = b)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: Assignment within condition of 'if' statement may indicate programming error
  if (a = c, b = c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Assignment within condition of 'if' statement may indicate programming error
}

void test_while(int a, int b, int c) {
  while (a = b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Assignment within condition of 'while' statement may indicate programming error
  while ((a = b)) {}
  while (a == b) {}

  while ((b > 0) ? (a = b) : c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: Assignment within condition of 'while' statement may indicate programming error
  while ((b > 0) ? c : (a = b)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: Assignment within condition of 'while' statement may indicate programming error
  while (a = b, b = c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: Assignment within condition of 'while' statement may indicate programming error
}

void test_do(int a, int b, int c) {
  do {} while (a = b);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: Assignment within condition of 'do' statement may indicate programming error
  do {} while ((a = b));
  do {} while (a == b);

  do {} while ((b > 0) ? (a = b) : c);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: Assignment within condition of 'do' statement may indicate programming error
  do {} while ((b > 0) ? c : (a = b));
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: Assignment within condition of 'do' statement may indicate programming error
}

void test_for(int a, int b, int c) {
  for (int i = 0; a = b; i++) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: Assignment within condition of 'for' statement may indicate programming error
  for (int i = 0; (a = b); i++) {}
  for (int i = 0; a == b; i++) {}

  for (int i = 0; (b > 0) ? (a = b) : c; ++i) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: Assignment within condition of 'for' statement may indicate programming error
  for (int i = 0; (b > 0) ? c : (a = b); ++i) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: Assignment within condition of 'for' statement may indicate programming error
}

void test_conditional(int a, int b, int c, int d) {
  int c1 = (a = b) ? 1 : 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Assignment within condition of conditional operator may indicate programming error
  int c2 = ((a = b)) ? 1 : 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Assignment within condition of conditional operator may indicate programming error
  int c3 = (a == b) ? 1 : 2;

  if ((c ? (a = b) : d) ? 1 : -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Assignment within condition of conditional operator may indicate programming error
  while ((c ? d : (a = b)) ? 1 : -1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: Assignment within condition of conditional operator may indicate programming error

  int c4 = (c ? (a = b) : 2);
  int c5 = (c ? 2 : (a = b));
}

void test_bin_op(int a, int b, int c, int d) {
  int c1 = (a = b) && c;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Assignment within operand of a logical operator may indicate programming error
  int c2 = c || (a = b);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: Assignment within operand of a logical operator may indicate programming error
  int c3 = ((a = b) && c) || (c == b - a);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: Assignment within operand of a logical operator may indicate programming error
  int c4 = c || ((a = b));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: Assignment within operand of a logical operator may indicate programming error
  int c5 = (a = b) + 2;
  int c6 = ((a = b) + 2) && c;

}

int f(int);

void test_misc(int a, int b, int c, int d) {
  if ((a = c, b = c)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: Assignment within condition of 'if' statement may indicate programming error
  if (a = c, (b = c)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: Assignment within condition of 'if' statement may indicate programming error
  if ((a > 0) ? ((b < 0) ? (a = b) : (a = c)) : (a = d)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: Assignment within condition of 'if' statement may indicate programming error
  // CHECK-MESSAGES: :[[@LINE-2]]:41: warning: Assignment within condition of 'if' statement may indicate programming error
  // CHECK-MESSAGES: :[[@LINE-3]]:52: warning: Assignment within condition of 'if' statement may indicate programming error
  while (a = c, (b = c, c = d)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: Assignment within condition of 'while' statement may indicate programming error
  for (d = 0; a = c, b = c, ((a > 0) ? d == a : (d = b)); ++d) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: Assignment within condition of 'for' statement may indicate programming error
  do {} while ((a > 0) ? (a = c, b = c) : d);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: Assignment within condition of 'do' statement may indicate programming error
  if ((a = b) && (c = d)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: Assignment within operand of a logical operator may indicate programming error
  // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: Assignment within operand of a logical operator may indicate programming error
  if ((a ? (b = c) : d) && (d ? c : (b = a))) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Assignment within operand of a logical operator may indicate programming error
  // CHECK-MESSAGES: :[[@LINE-2]]:40: warning: Assignment within operand of a logical operator may indicate programming error
  if (f((a = b) ? c : d)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Assignment within condition of conditional operator may indicate programming error
}

void test_no_warning(int a, int b, int c) {
  if ((a = b) != 0) {}
  if (!(a = b)) {}
  if ((int)(a = b)) {}
  if ((a = b) + c > 0) {}
  if ((b > 0) ? (a == b) : c) {}
  if ((b > 0) ? c : (a == b)) {}
  if (a = c, b == c) {}

  int arr[10] = {0};
  if (f(a = b)) {}
  if (arr[c = a]) {};
}
