// RUN: %clang_cc1 -std=c++11 -verify %s

union U {
  int x = 0; // expected-note {{previous initialization is here}}
  union {};
  union {
    int z;
    int y = 1; // expected-error {{initializing multiple members of union}}
  };
};
