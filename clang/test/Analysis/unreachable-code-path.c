// RUN: %clang_cc1 -analyze -analyzer-checker=core,deadcode.DeadStores,alpha.deadcode.UnreachableCode -verify -analyzer-opt-analyze-nested-blocks -Wno-unused-value %s

extern void foo(int a);

// The first few tests are non-path specific - we should be able to find them

void test(unsigned a) {
  switch (a) {
    a += 5; // expected-warning{{never executed}}
  case 2:
    a *= 10;
  case 3:
    a %= 2;
  }
  foo(a);
}

void test2(unsigned a) {
 help:
  if (a > 0)
    return;
  if (a == 0)
    return;
  foo(a); // expected-warning{{never executed}}
  goto help;
}

void test3(unsigned a) {
  while(1);
  if (a > 5) { // expected-warning{{never executed}}
    return;
  }
}

// These next tests are path-sensitive

void test4() {
  int a = 5;

  while (a > 1)
    a -= 2;

  if (a > 1) {
    a = a + 56; // expected-warning{{never executed}}
  }

  foo(a);
}

extern void bar(char c);

void test5(const char *c) {
  foo(c[0]);

  if (!c) {
    bar(1); // expected-warning{{never executed}}
  }
}

// These next tests are false positives and should not generate warnings

void test6(const char *c) {
  if (c) return;
  if (!c) return;
  __builtin_unreachable(); // no-warning
}

// Compile-time constant false positives
#define CONSTANT 0
enum test_enum { Off, On };
void test7() {
  if (CONSTANT)
    return; // no-warning

  if (sizeof(int))
    return; // no-warning

  if (Off)
    return; // no-warning
}

void test8() {
  static unsigned a = 0;

  if (a)
    a = 123; // no-warning

  a = 5;
}

// Check for bugs where multiple statements are reported
void test9(unsigned a) {
  switch (a) {
    if (a) // expected-warning{{never executed}}
      foo(a + 5); // no-warning
    else          // no-warning
      foo(a);     // no-warning
    case 1:
    case 2:
      break;
    default:
      break;
  }
}

// Tests from flow-sensitive version
void test10() {
  goto c;
  d:
  goto e; // expected-warning {{never executed}}
  c: ;
  int i;
  return;
  goto b; // expected-warning {{never executed}}
  goto a; // expected-warning {{never executed}}
  b:
  i = 1; // no-warning
  a:
  i = 2;  // no-warning
  goto f;
  e:
  goto d;
  f: ;
}

// test11: we can actually end up in the default case, even if it is not
// obvious: there might be something wrong with the given argument.
enum foobar { FOO, BAR };
extern void error();
void test11(enum foobar fb) {
  switch (fb) {
    case FOO:
      break;
    case BAR:
      break;
    default:
      error(); // no-warning
      return;
      error(); // expected-warning {{never executed}}
  }
}
