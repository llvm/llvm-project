// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

struct emp_1 { // expected-warning {{empty struct is a GNU extension}}
};

union emp_2 { // expected-warning {{empty union is a GNU extension}}
};

struct emp_3 { // expected-warning {{struct without named members is a GNU extension}}
  int : 0;
};

union emp_4 { // expected-warning {{union without named members is a GNU extension}}
  int : 0;
};

struct emp_5 { // expected-warning {{struct without named members is a GNU extension}}
  int : 0;
  int : 0;
};

union emp_6 { // expected-warning {{union without named members is a GNU extension}}
  int : 0;
  int : 0;
};

struct nonamed_1 { // expected-warning {{struct without named members is a GNU extension}}
  int : 4;
};

union nonamed_2 { // expected-warning {{union without named members is a GNU extension}}
  int : 4;
};

struct nonamed_3 { // expected-warning {{struct without named members is a GNU extension}}
  int : 4;
  unsigned int : 4;
};

union nonamed_4 { // expected-warning {{union without named members is a GNU extension}}
  int : 4;
  unsigned int : 4;
};
