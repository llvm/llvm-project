// RUN: %clang_cc1 -std=c++11 -verify %s

union U {
  int x = 0; // expected-note {{previous initialization is here}}
  union {}; // expected-warning {{does not declare anything}}
  union {
    int z;
    int y = 1; // expected-error {{initializing multiple members of union}}
  };
};

namespace GH149985 {
  union X {
    enum {
      csize = 42,
      cs = sizeof(int) // expected-note {{previous declaration is here}}
    };
    struct {
      int data; // expected-note {{previous declaration is here}}
      union X *cs[csize] = {}; // expected-error {{member of anonymous struct redeclares}} expected-note {{previous initialization is here}}
    };
    struct {
      int data; // expected-error {{member of anonymous struct redeclares}}
      union X *ds[2] = {}; // expected-error {{initializing multiple members of union}}
    };
  };

  union U {
    int x; // expected-note {{previous declaration is here}}
    union {
      int x = {}; // expected-error {{member of anonymous union redeclares}} expected-note {{previous initialization is here}}
    };
    int y = {}; // expected-error {{initializing multiple members of union}}
  };
}
