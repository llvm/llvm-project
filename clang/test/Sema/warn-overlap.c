// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-overlap-compare %s

#define mydefine 2

enum Choices {
  CHOICE_0 = 0,
  CHOICE_1 = 1
};

enum Unchoices {
  UNCHOICE_0 = 0,
  UNCHOICE_1 = 1
};

void f(int x) {
  int y = 0;

  // > || <
  if (x > 2 || x < 1) { }
  if (x > 2 || x < 2) { }
  if (x != 2 || x != 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 2 || x < 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 0 || x < 0) { }

  if (x > 2 || x <= 1) { }
  if (x > 2 || x <= 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 2 || x <= 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  if (x >= 2 || x < 1) { }
  if (x >= 2 || x < 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x < 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  if (x >= 2 || x <= 1) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x <= 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x <= 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 0 || x <= 0) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  // > && <
  if (x > 2 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x < 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x < 3) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 0 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x > 2 && x <= 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x <= 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x <= 3) { }

  if (x >= 2 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x < 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x < 3) { }
  if (x >= 0 && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x >= 2 && x <= 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x <= 2) { }
  if (x >= 2 && x <= 3) { }

  // !=, ==, ..
  if (x != 2 || x != 3) { }  // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x != 2 || x < 3) { }   // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x == 2 && x == 3) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x == 2 && x > 3) { }   // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x == 3 && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (3 == x && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x == mydefine && x > 3) { }
  if (x == (mydefine + 1) && x > 3) { }

  if (x != CHOICE_0 || x != CHOICE_1) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x == CHOICE_0 && x == CHOICE_1) { } // expected-warning {{overlapping comparisons always evaluate to false}}

  // Don't warn if comparing x to different types
  if (x == CHOICE_0 && x == 1) { }
  if (x != CHOICE_0 || x != 1) { }

  // "Different types" includes different enums
  if (x == CHOICE_0 && x == UNCHOICE_1) { }
  if (x != CHOICE_0 || x != UNCHOICE_1) { }
}

void enums(enum Choices c) {
  if (c != CHOICE_0 || c != CHOICE_1) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (c == CHOICE_0 && c == CHOICE_1) { } // expected-warning {{overlapping comparisons always evaluate to false}}

  // Don't warn if comparing x to different types
  if (c == CHOICE_0 && c == 1) { }
  if (c != CHOICE_0 || c != 1) { }

  // "Different types" includes different enums
  if (c == CHOICE_0 && c == UNCHOICE_1) { }
  if (c != CHOICE_0 || c != UNCHOICE_1) { }
}

// Don't generate a warning here.
void array_out_of_bounds() {
  int x;
  int buffer[4];
  x = (-7 > 0) ? (buffer[-7]) : 0;
}

void bool_contexts(int x) {
  if (x > 4 || x < 10) {}
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  for (;x > 4 || x < 10;) {}
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  while (x > 4 || x < 10) {}
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  do {} while (x > 4 || x < 10);
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  x = (x > 4 || x < 10) ? 1 : 2;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  if ((void)5, x > 4 || x < 10) {}
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
}

void assignment(int x) {
  int a = x > 4 || x < 10;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  int b = x < 2 && x > 5;
  // expected-warning@-1{{overlapping comparisons always evaluate to false}}

  int c = x != 1 || x != 3;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  int d = x == 1 && x == 2;
  // expected-warning@-1{{overlapping comparisons always evaluate to false}}

  int e = x < 1 || x != 0;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
}

int returns(int x) {
  return x > 4 || x < 10;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  return x < 2 && x > 5;
  // expected-warning@-1{{overlapping comparisons always evaluate to false}}

  return x != 1 || x != 3;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
  return x == 1 && x == 2;
  // expected-warning@-1{{overlapping comparisons always evaluate to false}}

  return x < 1 || x != 0;
  // expected-warning@-1{{overlapping comparisons always evaluate to true}}
}
