// RUN: %clang_cc1 -fsyntax-only -verify -Wassign-enum %s
// rdar://11824807

typedef enum CCTestEnum
{
  One,
  Two=4,
  Three
} CCTestEnum;

CCTestEnum test = 50; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
CCTestEnum test1 = -50; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}

CCTestEnum foo(CCTestEnum r) {
  return 20; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
}

enum Test2 { K_zero, K_one };
enum Test2 test2(enum Test2 *t) {
  *t = 20; // expected-warning {{integer constant not in range of enumerated type 'enum Test2'}}
  return 10; // expected-warning {{integer constant not in range of enumerated type 'enum Test2'}}
}

int main() {
  CCTestEnum test = 1; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  test = 600; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(2); // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(-1); // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(4);
  foo(Two+1);
}

