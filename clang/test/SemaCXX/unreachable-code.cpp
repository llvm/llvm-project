// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -Wunreachable-code -fblocks -verify %s

int j;
void bar() { }
int test1() {
  for (int i = 0;
       i != 10;
       ++i) {  // expected-warning {{will never be executed}}
    if (j == 23) // missing {}'s
      bar();
      return 1;
  }
  return 0;
  return 1;    // expected-warning {{will never be executed}}
}

void test2(int i) {
  switch (i) {
  case 0:
    break;
    bar();     // expected-warning {{will never be executed}}
  case 2:
    switch (i) {
    default:
    a: goto a;
    }
    bar();     // expected-warning {{will never be executed}}
  }
  b: goto b;
  bar();       // expected-warning {{will never be executed}}
}

void test3() {
  ^{ return;
     bar();    // expected-warning {{will never be executed}}
  }();
  while (++j) {
    continue;
    bar();     // expected-warning {{will never be executed}}
  }
}

// PR 6130 - Don't warn about bogus unreachable code with throw's and
// temporary objects.
class PR6130 {
public:
  PR6130();
  ~PR6130();
};

int pr6130(unsigned i) {
  switch(i) {
    case 0: return 1;
    case 1: return 2;
    default:
      throw PR6130(); // no-warning
  }
}
