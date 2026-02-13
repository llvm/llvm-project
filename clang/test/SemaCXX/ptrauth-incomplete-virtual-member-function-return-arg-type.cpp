// RUN: %clang_cc1 -triple arm64-apple-ios   -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics -fptrauth-calls %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics -fptrauth-calls %s

struct Incomplete0; // expected-note 3 {{forward declaration of 'Incomplete0'}}

template <class T>
struct Incomplete1; // expected-note {{template is declared here}}

struct Complete0 {
};

template <class T>
struct Complete1 {
};

struct S {
  virtual int foo();
  virtual Incomplete0 virtual0(); // expected-note 2 {{'Incomplete0' is incomplete}}
  virtual void virtual1(Incomplete1<int>); // expected-note {{'Incomplete1<int>' is incomplete}}
  virtual Complete0 virtual2();
  virtual Complete1<int> virtual3();
  Incomplete0 nonvirtual0();
  template <class T>
  void m0() {
    (void)&S::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual}}
  }
};

template <bool T>
struct S2 {
  virtual Incomplete0 virtual0() noexcept(T); // expected-note {{'Incomplete0' is incomplete}}

  void m0() {
    (void)&S2<T>::virtual0;
  }

  void m1() {
    (void)&S2<T>::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual}}
  }
};

void test_incomplete_virtual_member_function_return_arg_type() {
  (void)&S::virtual0; // expected-error {{incomplete type 'Incomplete0}} expected-note {{cannot take an address of a virtual member function}}
  (void)&S::virtual1; // expected-error {{implicit instantiation of undefined template 'Incomplete1<int>'}} expected-note {{cannot take an address of a virtual member function}}
  (void)&S::virtual2;
  (void)&S::virtual3;
  (void)&S::nonvirtual0;
  int s = sizeof(&S::virtual0);
  S2<true>().m1(); // expected-note {{in instantiation of}}
}

