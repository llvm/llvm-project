// RUN: %clang_cc1 -triple arm64-apple-ios   -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics -fptrauth-calls %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics -fptrauth-calls %s

struct Incomplete0; // expected-note 7 {{forward declaration of 'Incomplete0'}}

template <class T>
struct Incomplete1;

template <class T>
struct DoNotInstantiate {
  static_assert(sizeof(T) == 0, "should not be instantiated");
};

template <class T>
struct MustInstantiateReturn {
  static_assert(sizeof(T) == 0, "return type was instantiated"); // expected-error {{static assertion failed due to requirement 'sizeof(int) == 0': return type was instantiated}} expected-note {{expression evaluates to}}
};

struct Complete0 {
};

template <class T>
struct Complete1 {
};

struct S {
  virtual int foo();
  virtual Incomplete0 virtual0(); // expected-note 4 {{'Incomplete0' is incomplete}}
  virtual void virtual1(Incomplete1<int>);
  virtual Complete0 virtual2();
  virtual Complete1<int> virtual3();
  virtual MustInstantiateReturn<int> virtual4(); // expected-note {{'MustInstantiateReturn<int>' is incomplete}}
  virtual DoNotInstantiate<int> *virtual4_ptr();
  virtual void virtual5(DoNotInstantiate<int>);
  virtual Incomplete0 overloaded(int); // expected-note {{'Incomplete0' is incomplete}}
  virtual void overloaded(Incomplete0);
  virtual void swift_incomplete(Incomplete0) __attribute__((swiftasynccall)); // expected-note {{'Incomplete0' is incomplete}}
  virtual void swift_complete(Complete0) __attribute__((swiftasynccall));
  Incomplete0 nonvirtual0();
  template <class T>
  void m0() {
    (void)&S::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
    (void)&S::virtual1;
  }
};

template <bool T>
struct S2 {
  virtual Incomplete0 virtual0() noexcept(T); // expected-note {{'Incomplete0' is incomplete}}

  void m0() {
    (void)&S2<T>::virtual0;
  }

  void m1() {
    (void)&S2<T>::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
  }
};

template <auto P>
struct NTTP {};

constexpr auto ce_virtual0 = &S::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
constexpr auto ce_virtual1 = &S::virtual1;
NTTP<&S::virtual0> nttp0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
NTTP<&S::virtual1> nttp1;
NTTP<&S::virtual4_ptr> nttp4_ptr;
NTTP<&S::virtual5> nttp5;

void test_incomplete_virtual_member_function_return_arg_type() {
  (void)&S::virtual0; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
  (void)&S::virtual1;
  (void)&S::virtual2;
  (void)&S::virtual3;
  (void)&S::virtual4; // expected-note {{in instantiation of template class 'MustInstantiateReturn<int>' requested here}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
  (void)&S::virtual4_ptr;
  (void)&S::virtual5;
  void (S::*ov_ok)(Incomplete0) = &S::overloaded;
  Incomplete0 (S::*ov_err)(int) = &S::overloaded; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its return type is incomplete}}
  (void)&S::swift_incomplete; // expected-error {{incomplete type 'Incomplete0'}} expected-note {{cannot take an address of a virtual member function if its parameter type is incomplete}}
  (void)&S::swift_complete;
  (void)&S::nonvirtual0;
  int s = sizeof(&S::virtual0);
  S().m0<int>();
  S2<true>().m1(); // expected-note {{in instantiation of}}
}
