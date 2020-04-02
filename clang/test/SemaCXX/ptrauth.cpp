// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++17 -fsyntax-only -verify -fptrauth-intrinsics -fptrauth-calls %s

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

template <class T>
constexpr unsigned dependentOperandDisc() {
  return __builtin_ptrauth_type_discriminator(T);
}

void test_builtin_ptrauth_type_discriminator(unsigned s) {
  typedef int (S::*MemFnTy)();
  MemFnTy memFnPtr;
  int (S::*memFnPtr2)();

  constexpr unsigned d = __builtin_ptrauth_type_discriminator(MemFnTy);
  static_assert(d == 60844);
  static_assert(__builtin_ptrauth_type_discriminator(int (S::*)()) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr)) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(memFnPtr2)) == d);
  static_assert(__builtin_ptrauth_type_discriminator(decltype(&S::foo)) == d);
  static_assert(dependentOperandDisc<decltype(&S::foo)>() == d);
  static_assert(__builtin_ptrauth_type_discriminator(void (S::*)(int)) == 39121);
  static_assert(__builtin_ptrauth_type_discriminator(void (S::*)(float)) == 52453);
  static_assert(__builtin_ptrauth_type_discriminator(int *) == 42396);

  int t;
  int vmarray[s];
  __builtin_ptrauth_type_discriminator(t); // expected-error {{unknown type name 't'}}
  __builtin_ptrauth_type_discriminator(&t); // expected-error {{expected a type}}
  __builtin_ptrauth_type_discriminator(decltype(vmarray)); // expected-error {{cannot pass variably-modified type 'decltype(vmarray)'}}
}

void test_incomplete_virtual_member_function_return_arg_type() {
  (void)&S::virtual0; // expected-error {{incomplete type 'Incomplete0}} expected-note {{cannot take an address of a virtual member function}}
  (void)&S::virtual1; // expected-error {{implicit instantiation of undefined template 'Incomplete1<int>'}} expected-note {{cannot take an address of a virtual member function}}
  (void)&S::virtual2;
  (void)&S::virtual3;
  (void)&S::nonvirtual0;
  int s = sizeof(&S::virtual0);
  S2<true>().m1(); // expected-note {{in instantiation of}}
}
