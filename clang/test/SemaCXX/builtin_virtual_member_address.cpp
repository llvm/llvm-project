// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Base {
  virtual void func1();
  virtual void func2();

  virtual void overloaded();
  virtual void overloaded() const;

  void nonvirt();
};

struct Derived : Base {
  virtual void func1();
};

struct Unrelated {
  virtual void func1();
};

void *simple(Base &b) {
  return __builtin_virtual_member_address(b, &Base::func1);
}

void test(Base &b, Derived &d, Unrelated &u) {
  __builtin_virtual_member_address(42, &Base::func1); // expected-error {{first argument to __builtin_virtual_member_address must have C++ class type}}
  __builtin_virtual_member_address(u, &Base::func1); // expected-error {{first argument to __builtin_virtual_member_address must have a type deriving from class where second argument was defined}}

  __builtin_virtual_member_address(b, &Derived::func1); // expected-error {{first argument to __builtin_virtual_member_address must have a type deriving from class where second argument was defined}}
  __builtin_virtual_member_address(b, &Unrelated::func1); // expected-error {{first argument to __builtin_virtual_member_address must have a type deriving from class where second argument was defined}}

  __builtin_virtual_member_address(b, 42); // expected-error {{second argument to __builtin_virtual_member_address must be the address of a virtual C++ member function: for example '&Foo::func'}}
  __builtin_virtual_member_address(b, &Base::nonvirt); // expected-error {{second argument to __builtin_virtual_member_address must be the address of a virtual C++ member function: for example '&Foo::func'}}

  (void)__builtin_virtual_member_address(d, &Base::overloaded); // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}} expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}}
  (void)__builtin_virtual_member_address(d,(void (Base::*)() const) &Base::overloaded);

  (void)__builtin_virtual_member_address(d, &Base::func1);
  (void)__builtin_virtual_member_address(Base(), &Base::func1);
}

template<typename T> void test(T &t, Base &b, Derived &d) {
  (void)__builtin_virtual_member_address(&b, &Base::func1);
  (void)__builtin_virtual_member_address(t, &Base::func1);
  (void)__builtin_virtual_member_address(&t, &Base::func1);
  (void)__builtin_virtual_member_address(t, &T::func1);
  (void)__builtin_virtual_member_address(&t, &T::func1);
  (void)__builtin_virtual_member_address(d, &T::func1);
  (void)__builtin_virtual_member_address(&d, &T::func1);
  (void)__builtin_virtual_member_address(t, &Unrelated::func1); // expected-error {{first argument to __builtin_virtual_member_address must have a type deriving from class where second argument was defined}} expected-error {{first argument to __builtin_virtual_member_address must have a type deriving from class where second argument was defined}}

}

void foo() {
  Base b;
  Derived d;

  test<Base>(b, b, d); // expected-note {{in instantiation of function template specialization 'test<Base>' requested here}}
  test<Derived>(d, b, d); // expected-note {{in instantiation of function template specialization 'test<Derived>' requested here}}
}
