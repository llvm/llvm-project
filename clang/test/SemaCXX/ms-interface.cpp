// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -Wno-microsoft -std=c++11

__interface I1 {
  // expected-error@+1 {{user-declared constructor is not permitted within an interface type}}
  I1();
  // expected-error@+1 {{user-declared destructor is not permitted within an interface type}}
  ~I1();
  virtual void fn1() const;
  // expected-error@+1 {{operator 'operator!' is not permitted within an interface type}}
  bool operator!();
  // expected-error@+1 {{operator 'operator int' is not permitted within an interface type}}
  operator int();
  // expected-error@+1 {{nested class I1::<anonymous> is not permitted within an interface type}}
  struct { int a; };
  void fn2() {
    struct A { }; // should be ignored: not a nested class
  }
protected: // expected-error {{interface types cannot specify 'protected' access}}
  typedef void void_t;
  using int_t = int;
private:   // expected-error {{interface types cannot specify 'private' access}}
  static_assert(true, "oops");
};

__interface I2 {
  // expected-error@+1 {{data member 'i' is not permitted within an interface type}}
  int i;
  // expected-error@+1 {{static member function 'fn1' is not permitted within an interface type}}
  static int fn1();
private:   // expected-error {{interface types cannot specify 'private' access}}
  // expected-error@+1 {{non-public member function 'fn2' is not permitted within an interface type}}
  void fn2();
protected: // expected-error {{interface types cannot specify 'protected' access}}
  // expected-error@+1 {{non-public member function 'fn3' is not permitted within an interface type}}
  void fn3();
public:
  void fn4();
};

// expected-error@+1 {{'final' keyword not permitted with interface types}}
__interface I3 final {
};

__interface I4 : I1, I2 {
  void fn1() const override;
  // expected-error@+1 {{'final' keyword not permitted with interface types}}
  void fn2() final;
};

// expected-error@+1 {{interface type cannot inherit from non-public 'interface I1'}}
__interface I5 : private I1 {
};

template <typename X>
__interface I6 : X {
};

struct S { };
class C { };
__interface I { };

static_assert(!__is_interface_class(S), "oops");
static_assert(!__is_interface_class(C), "oops");
static_assert(__is_interface_class(I), "oops");

// expected-error@55 {{interface type cannot inherit from 'struct S'}}
// expected-note@+1 {{in instantiation of template class 'I6<S>' requested here}}
struct S1 : I6<S> {
};

// expected-error@55 {{interface type cannot inherit from 'class C'}}
// expected-note@+1 {{in instantiation of template class 'I6<C>' requested here}}
class C1 : I6<C> {
};

class C2 : I6<I> {
};
