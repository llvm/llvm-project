// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fcxx-exceptions %s

int i = delete; // expected-error {{only functions can have deleted definitions}}

void fn() = delete; // expected-note {{candidate function has been explicitly deleted}}

void fn2(); // expected-note {{previous declaration is here}}
void fn2() = delete; // expected-error {{deleted definition must be first declaration}}

void fn3() = delete; // expected-note {{previous definition is here}}
void fn3() { // expected-error {{redefinition}}
}

void ov(int) {} // expected-note {{candidate function}}
void ov(double) = delete; // expected-note {{candidate function has been explicitly deleted}}

struct WithDel {
  WithDel() = delete; // expected-note {{function has been explicitly marked deleted here}}
  void fn() = delete; // expected-note {{function has been explicitly marked deleted here}}
  operator int() = delete; // expected-note {{function has been explicitly marked deleted here}}
  void operator +(int) = delete;

  int i = delete; // expected-error {{only functions can have deleted definitions}}
};

void test() {
  fn(); // expected-error {{call to deleted function 'fn'}}
  ov(1);
  ov(1.0); // expected-error {{call to deleted function 'ov'}}

  WithDel dd; // expected-error {{call to deleted constructor of 'WithDel'}}
  WithDel *d = 0;
  d->fn(); // expected-error {{attempt to use a deleted function}}
  int i = *d; // expected-error {{invokes a deleted function}}
}

struct DelDtor {
  ~DelDtor() = delete; // expected-note 9{{here}}
};
void f() {
  DelDtor *p = new DelDtor[3]; // expected-error {{attempt to use a deleted function}}
  delete [] p; // expected-error {{attempt to use a deleted function}}
  const DelDtor &dd2 = DelDtor(); // expected-error {{attempt to use a deleted function}}
  DelDtor dd; // expected-error {{attempt to use a deleted function}}
  throw dd; // expected-error {{attempt to use a deleted function}}
}
struct X : DelDtor {
  ~X() {} // expected-error {{attempt to use a deleted function}}
};
struct Y {
  DelDtor dd;
  ~Y() {} // expected-error {{attempt to use a deleted function}}
};
struct Z : virtual DelDtor {
  ~Z() {} // expected-error {{attempt to use a deleted function}}
};
DelDtor dd; // expected-error {{attempt to use a deleted function}}

template<typename> void test2() = delete;
template void test2<int>();

template<typename> void test3() = delete;
template<typename> void test3();
template void test3<int>();

void test4() {} // expected-note {{previous definition is here}}
void test4() = delete; // expected-error {{redefinition of 'test4'}}
