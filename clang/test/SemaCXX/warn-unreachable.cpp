// RUN: %clang_cc1 %s -fcxx-exceptions -fexceptions -fsyntax-only -verify -fblocks -Wunreachable-code -Wno-unused-value

int &halt() __attribute__((noreturn));
int &live();
int dead();
int liveti() throw(int);
int (*livetip)() throw(int);

int test1() {
  try {
    live();
  } catch (int i) {
    live();
  }
  return 1;
}

void test2() {
  try {
    live();
  } catch (int i) {
    live();
  }
  try {
    liveti();
  } catch (int i) {
    live();
  }
  try {
    livetip();
  } catch (int i) {
    live();
  }
  throw 1;
  dead();       // expected-warning {{will never be executed}}
}


void test3() {
  halt()
    --;         // expected-warning {{will never be executed}}
  // FIXME: The unreachable part is just the '?', but really all of this
  // code is unreachable and shouldn't be separately reported.
  halt()        // expected-warning {{will never be executed}}
    ? 
    dead() : dead();
  live(),
    float       
      (halt()); // expected-warning {{will never be executed}}
}

void test4() {
  struct S {
    int mem;
  } s;
  S &foor();
  halt(), foor()// expected-warning {{will never be executed}}
    .mem;       
}

void test5() {
  struct S {
    int mem;
  } s;
  S &foor() __attribute__((noreturn));
  foor()
    .mem;       // expected-warning {{will never be executed}}
}

void test6() {
  struct S {
    ~S() { }
    S(int i) { }
  };
  live(),
    S
      (halt());  // expected-warning {{will never be executed}}
}

// Don't warn about unreachable code in template instantiations, as
// they may only be unreachable in that specific instantiation.
void isUnreachable();

template <typename T> void test_unreachable_templates() {
  T::foo();
  isUnreachable();  // no-warning
}

struct TestUnreachableA {
  static void foo() __attribute__((noreturn));
};
struct TestUnreachableB {
  static void foo();
};

void test_unreachable_templates_harness() {
  test_unreachable_templates<TestUnreachableA>();
  test_unreachable_templates<TestUnreachableB>(); 
}

// Do warn about explict template specializations, as they represent
// actual concrete functions that somebody wrote.

template <typename T> void funcToSpecialize() {}
template <> void funcToSpecialize<int>() {
  halt();
  dead(); // expected-warning {{will never be executed}}
}

