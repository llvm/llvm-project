// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -Wunreachable-code-aggressive -fblocks -verify %s

int j;
int bar();
int test1() {
  for (int i = 0;
       i != 10;
       ++i) {  // expected-warning {{loop will run at most once (loop increment never executed)}}
    if (j == 23) // missing {}'s
      bar();
      return 1;
  }
  return 0;
  return 1; // expected-warning {{will never be executed}}
}

int test1_B() {
  for (int i = 0;
       i != 10;
       ++i) {  // expected-warning {{loop will run at most once (loop increment never executed)}}
    if (j == 23) // missing {}'s
      bar();
      return 1;
  }
  return 0;
  return bar(); // expected-warning {{will never be executed}}
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

extern "C" void foo(void);
extern "C" __attribute__((weak)) decltype(foo) foo;

void weak_redecl() {
  if (foo)
    return;
  bar(); // no-warning
}

namespace pr52103 {

void g(int a);

void f(int a) {
  if (a > 4) [[ likely ]] { // no-warning
    return;
  }

  if (a > 4) [[ unlikely ]] { // no-warning
    return;

    return; // expected-warning {{will never be executed}}
  }

  [[clang::musttail]] return g(a); // no-warning

  [[clang::musttail]] return g(a); // expected-warning {{will never be executed}}
}

}
