// RUN: %clang_analyze_cc1 -analyzer-checker=core -Wno-uninitialized -verify %s

int if_cond(void) {
  int foo;
  if (foo) //expected-warning {{Branch condition evaluates to a garbage value}}
    return 1;
  return 2;
}

int logical_op_and_if_cond(void) {
  int foo, bar;
  if (foo && bar) //expected-warning {{Branch condition evaluates to a garbage value}}
    return 1;
  return 2;
}

int logical_op_cond(int arg) {
  int foo;
  if (foo && arg) //expected-warning {{Branch condition evaluates to a garbage value}}
    return 1;
  return 2;
}

int if_cond_after_logical_op(int arg) {
  int foo;
  if (arg && foo) //expected-warning {{Branch condition evaluates to a garbage value}}
    return 1;
  return 2;
}

int ternary_cond(void) {
  int foo;
  return foo ? 1 : 2; //expected-warning {{Branch condition evaluates to a garbage value}}
}

int while_cond(void) {
  int foo;
  while (foo) //expected-warning {{Branch condition evaluates to a garbage value}}
    return 1;
  return 2;
}

int do_while_cond(void) {
  int foo, bar;
  do {
    foo = 43;
  } while (bar); //expected-warning {{Branch condition evaluates to a garbage value}}
  return foo;
}

int switch_cond(void) {
  int foo;
  switch (foo) { //expected-warning {{Branch condition evaluates to a garbage value}}
    case 1:
      return 3;
    case 2:
      return 440;
    default:
      return 6772;
  }
}
