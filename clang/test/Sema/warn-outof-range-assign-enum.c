// RUN: %clang_cc1 -fsyntax-only -verify -Wassign-enum %s

typedef enum CCTestEnum
{
  One,
  Two=4,
  Three
} CCTestEnum;

CCTestEnum test = 50; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
CCTestEnum test1 = -50; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}

// Explicit cast should silence the warning.
static const CCTestEnum SilenceWithCast1 = 51; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
static const CCTestEnum SilenceWithCast2 = (CCTestEnum) 51; // no-warning
static const CCTestEnum SilenceWithCast3 = (const CCTestEnum) 51; // no-warning
static const CCTestEnum SilenceWithCast4 = (const volatile CCTestEnum) 51; // no-warning

void SilenceWithCastLocalVar(void) {
  CCTestEnum SilenceWithCast1 = 51; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  CCTestEnum SilenceWithCast2 = (CCTestEnum) 51; // no-warning
  CCTestEnum SilenceWithCast3 = (const CCTestEnum) 51; // no-warning
  CCTestEnum SilenceWithCast4 = (const volatile CCTestEnum) 51; // no-warning

  const CCTestEnum SilenceWithCast1c = 51; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  const CCTestEnum SilenceWithCast2c = (CCTestEnum) 51; // no-warning
  const CCTestEnum SilenceWithCast3c = (const CCTestEnum) 51; // no-warning
  const CCTestEnum SilenceWithCast4c = (const volatile CCTestEnum) 51; // no-warning
}

CCTestEnum foo(CCTestEnum r) {
  return 20; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
}

enum Test2 { K_zero, K_one };
enum Test2 test2(enum Test2 *t) {
  *t = 20; // expected-warning {{integer constant not in range of enumerated type 'enum Test2'}}
  return 10; // expected-warning {{integer constant not in range of enumerated type 'enum Test2'}}
}

// PR15069
typedef enum
{
  a = 0
} T;

void f(void)
{
  T x = a;
  x += 1; // expected-warning {{integer constant not in range of enumerated type}}
}

int main(void) {
  CCTestEnum test = 1; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  test = 600; // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(2); // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(-1); // expected-warning {{integer constant not in range of enumerated type 'CCTestEnum'}}
  foo(4);
  foo(Two+1);
}

