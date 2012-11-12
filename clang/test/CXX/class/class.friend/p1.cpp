// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Outer {
  struct Inner {
    int intfield;
  };
};

struct Base {
  void base_member();
  
  typedef int Int;
  Int typedeffed_member();
};

struct Derived : public Base {
};

int myglobal;

void global_function();
extern "C" {
  void global_c_function();
}

class A {
  class AInner {
  };

  friend class PreDeclared;
  friend class Outer::Inner;
  friend int Outer::Inner::intfield; // expected-error {{friends can only be classes or functions}}
  friend int Outer::Inner::missing_field; //expected-error {{friends can only be classes or functions}}
  friend int myoperation(float); // okay
  friend int myglobal;   // expected-error {{friends can only be classes or functions}}

  friend void global_function();
  friend void global_c_function();

  friend class UndeclaredSoFar;
  UndeclaredSoFar x; // expected-error {{unknown type name 'UndeclaredSoFar'}}

  void a_member();
  friend void A::a_member(); // expected-error {{friends cannot be members of the declaring class}}
  friend void a_member(); // okay (because we ignore class scopes when looking up friends)
  friend class A::AInner; // this is okay as an extension
  friend class AInner; // okay, refers to ::AInner

  friend void Derived::missing_member(); // expected-error {{no function named 'missing_member' with type 'void ()' was found in the specified scope}}

  friend void Derived::base_member(); // expected-error {{no function named 'base_member' with type 'void ()' was found in the specified scope}}

  friend int Base::typedeffed_member(); // okay: should look through typedef

  // These test that the friend is properly not being treated as a
  // member function.
  friend A operator|(const A& l, const A& r); // okay
  friend A operator|(const A& r); // expected-error {{overloaded 'operator|' must be a binary operator (has 1 parameter)}}

  friend operator bool() const; // expected-error {{must use a qualified name when declaring a conversion operator as a friend}} \
       // expected-error{{non-member function cannot have 'const' qualifier}}

  typedef void ftypedef();
  friend ftypedef typedeffed_function; // okay (because it's not declared as a member)

  class facet;
  friend class facet;  // should not assert
  class facet {};
};

A::UndeclaredSoFar y; // expected-error {{no type named 'UndeclaredSoFar' in 'A'}}

class PreDeclared;

int myoperation(float f) {
  return (int) f;
}
