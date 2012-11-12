// RUN: %clang_cc1 -fsyntax-only -verify %s 

void abort() __attribute__((noreturn));

class Okay {
  int a_;
};

class Virtual {
  virtual void foo() { abort(); } // expected-note 4 {{because type 'Virtual' has a virtual member function}}
};

class VirtualBase : virtual Okay { // expected-note 4 {{because type 'VirtualBase' has a virtual base class}}
};

class Ctor {
  Ctor() { abort(); } // expected-note 4 {{because type 'Ctor' has a user-declared constructor}}
};
class Ctor2 {
  Ctor2(); // expected-note 3 {{because type 'Ctor2' has a user-declared constructor}}
};
class CtorTmpl {
  template<typename T> CtorTmpl(); // expected-note {{because type 'CtorTmpl' has a user-declared constructor}}
};

class CopyCtor {
  CopyCtor(CopyCtor &cc) { abort(); } // expected-note 4 {{because type 'CopyCtor' has a user-declared copy constructor}}
};

class CopyAssign {
  CopyAssign& operator=(CopyAssign& CA) { abort(); } // expected-note 4 {{because type 'CopyAssign' has a user-declared copy assignment operator}}
};

class Dtor {
  ~Dtor() { abort(); } // expected-note 4 {{because type 'Dtor' has a user-declared destructor}}
};

union U1 {
  Virtual v; // expected-error {{union member 'v' has a non-trivial copy constructor}}
  VirtualBase vbase; // expected-error {{union member 'vbase' has a non-trivial copy constructor}}
  Ctor ctor; // expected-error {{union member 'ctor' has a non-trivial constructor}}
  Ctor2 ctor2; // expected-error {{union member 'ctor2' has a non-trivial constructor}}
  CtorTmpl ctortmpl; // expected-error {{union member 'ctortmpl' has a non-trivial constructor}}
  CopyCtor copyctor; // expected-error {{union member 'copyctor' has a non-trivial copy constructor}}
  CopyAssign copyassign; // expected-error {{union member 'copyassign' has a non-trivial copy assignment operator}}
  Dtor dtor; // expected-error {{union member 'dtor' has a non-trivial destructor}}
  Okay okay;
};

union U2 {
  struct {
    Virtual v; // expected-note {{because type 'U2::<anonymous struct}}
  } m1; // expected-error {{union member 'm1' has a non-trivial copy constructor}}
  struct {
    VirtualBase vbase; // expected-note {{because type 'U2::<anonymous struct}}
  } m2; // expected-error {{union member 'm2' has a non-trivial copy constructor}}
  struct {
    Ctor ctor; // expected-note {{because type 'U2::<anonymous struct}}
  } m3; // expected-error {{union member 'm3' has a non-trivial constructor}}
  struct {
    Ctor2 ctor2; // expected-note {{because type 'U2::<anonymous struct}}
  } m3a; // expected-error {{union member 'm3a' has a non-trivial constructor}}
  struct {
    CopyCtor copyctor; // expected-note {{because type 'U2::<anonymous struct}}
  } m4; // expected-error {{union member 'm4' has a non-trivial copy constructor}}
  struct {
    CopyAssign copyassign; // expected-note {{because type 'U2::<anonymous struct}}
  } m5; // expected-error {{union member 'm5' has a non-trivial copy assignment operator}}
  struct {
    Dtor dtor; // expected-note {{because type 'U2::<anonymous struct}}
  } m6; // expected-error {{union member 'm6' has a non-trivial destructor}}
  struct {
    Okay okay;
  } m7;
};

union U3 {
  struct s1 : Virtual { // expected-note {{because type 'U3::s1' has a base class with a non-trivial copy constructor}}
  } m1; // expected-error {{union member 'm1' has a non-trivial copy constructor}}
  struct s2 : VirtualBase { // expected-note {{because type 'U3::s2' has a base class with a non-trivial copy constructor}}
  } m2; // expected-error {{union member 'm2' has a non-trivial copy constructor}}
  struct s3 : Ctor { // expected-note {{because type 'U3::s3' has a base class with a non-trivial constructor}}
  } m3; // expected-error {{union member 'm3' has a non-trivial constructor}}
  struct s3a : Ctor2 { // expected-note {{because type 'U3::s3a' has a base class with a non-trivial constructor}}
  } m3a; // expected-error {{union member 'm3a' has a non-trivial constructor}}
  struct s4 : CopyCtor { // expected-note {{because type 'U3::s4' has a base class with a non-trivial copy constructor}}
  } m4; // expected-error {{union member 'm4' has a non-trivial copy constructor}}
  struct s5 : CopyAssign { // expected-note {{because type 'U3::s5' has a base class with a non-trivial copy assignment operator}}
  } m5; // expected-error {{union member 'm5' has a non-trivial copy assignment operator}}
  struct s6 : Dtor { // expected-note {{because type 'U3::s6' has a base class with a non-trivial destructor}}
  } m6; // expected-error {{union member 'm6' has a non-trivial destructor}}
  struct s7 : Okay {
  } m7;
};

union U4 {
  static int i1; // expected-warning {{static data member 'i1' in union is a C++11 extension}}
};
int U4::i1 = 10;

union U5 {
  int& i1; // expected-error {{union member 'i1' has reference type 'int &'}}
};

template <class A, class B> struct Either {
  bool tag;
  union { // expected-note 6 {{in instantiation of member class}}
    A a;
    B b; // expected-error 6 {{non-trivial}}
  };

  Either(const A& a) : tag(true), a(a) {}
  Either(const B& b) : tag(false), b(b) {}
};

void fred() {
  Either<int,Virtual> virt(0); // expected-note {{in instantiation of template}}
  Either<int,VirtualBase> vbase(0); // expected-note {{in instantiation of template}}
  Either<int,Ctor> ctor(0); // expected-note {{in instantiation of template}}
  Either<int,CopyCtor> copyctor(0); // expected-note {{in instantiation of template}}
  Either<int,CopyAssign> copyassign(0); // expected-note {{in instantiation of template}}
  Either<int,Dtor> dtor(0); // expected-note {{in instantiation of template}}
  Either<int,Okay> okay(0);
}
