// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify -Wswitch %s

typedef enum : long { E0 } E;
void test1(E e) {
  auto v = E0;
  switch (v) { } // expected-warning {{enumeration value 'E0' not handled in switch}}
}

void test2(E e) {
  __auto_type v = E0;
  switch (v) { } // expected-warning {{enumeration value 'E0' not handled in switch}}
}

void test3(_Bool b, E e) {
  __auto_type v = E0;
  if (b) {
    v = e;
  }
  switch (v) { } // expected-warning {{enumeration value 'E0' not handled in switch}}
}
