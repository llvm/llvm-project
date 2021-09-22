// RUN: %clang_cc1 -verify -ffixed-point %s

union a {
  _Accum x;
  int i;
};

int fn1() {
  union a m;
  m.x = 5.6k;
  return m.i;
}

int fn2() {
  union a m;
  m.x = 7, 5.6k; // expected-warning {{expression result unused}}
  return m.x, m.i; // expected-warning {{left operand of comma operator has no effect}}
}

_Accum acc = (0.5r, 6.9k); // expected-warning {{left operand of comma operator has no effect}}
