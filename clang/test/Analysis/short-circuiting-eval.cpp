// RUN: %clang_analyze_cc1 -analyzer-checker=core.DivideZero -verify %s

int div0LogicalOpInTernary(bool b1) {
  int y = (b1 || b1) ? 0 : 1;
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0LogicalAndArith(bool b1, int x) {
  int y = (b1 || (x < 3)) ? 0 : 1;
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0NestedLogicalOp(bool b1) {
  int y = (b1 && b1 || b1 && b1) ? 0 : 1;
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0TernaryInTernary(bool b) {
  int y = ((b || b) ? false : true) ? 0 : 1;
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0LogicalOpParensInTernary(bool b1) {
  int y = ((((b1)) || ((b1)))) ? 0 : 1;
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0LogicalOpInsideStExpr(bool b1) {
  int y = ({1; (b1 || b1);}) ? 0 : 1;
  // expected-warning@-1 {{expression result unused}}
  return 1 / y; // expected-warning {{Division by zero}}
}

int div0StExprInsideLogicalOp(bool b1) {
  int y = (({1; b1;}) || ({1; b1;})) ? 0 : 1;
  // expected-warning@-1 {{expression result unused}}
  // expected-warning@-2 {{expression result unused}}
  return 1 / y; // expected-warning {{Division by zero}}
}
