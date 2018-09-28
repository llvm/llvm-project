// RUN: %clang_cc1 -verify -std=c++11 %s -Wno-defaulted-function-deleted

struct NonTrivDtor {
  ~NonTrivDtor();
};
struct DeletedDtor {
  ~DeletedDtor() = delete; // expected-note 5 {{deleted here}}
};
class InaccessibleDtor {
  ~InaccessibleDtor() = default;
};

// A defaulted destructor for a class X is defined as deleted if:

// -- X is a union-like class that has a variant member with a non-trivial
// destructor.
union A1 {
  A1();
  NonTrivDtor n; // expected-note {{destructor of 'A1' is implicitly deleted because variant field 'n' has a non-trivial destructor}}
};
A1 a1; // expected-error {{deleted function}}
struct A2 {
  A2();
  union {
    NonTrivDtor n; // expected-note {{because variant field 'n' has a non-trivial destructor}}
  };
};
A2 a2; // expected-error {{deleted function}}
union A3 {
  A3();
  NonTrivDtor n[3]; // expected-note {{because variant field 'n' has a non-trivial destructor}}
};
A3 a3; // expected-error {{deleted function}}
struct A4 {
  A4();
  union {
    NonTrivDtor n[3]; // expected-note {{because variant field 'n' has a non-trivial destructor}}
  };
};
A4 a4; // expected-error {{deleted function}}

// -- any of the non-static data members has class type M (or array thereof) and
// M has a deleted or inaccessible destructor.
struct B1 {
  B1();
  DeletedDtor a; // expected-note {{because field 'a' has a deleted destructor}}
};
B1 b1; // expected-error {{deleted function}}
struct B2 {
  B2();
  InaccessibleDtor a; // expected-note {{because field 'a' has an inaccessible destructor}}
};
B2 b2; // expected-error {{deleted function}}
struct B3 {
  B3();
  DeletedDtor a[4]; // expected-note {{because field 'a' has a deleted destructor}}
};
B3 b3; // expected-error {{deleted function}}
struct B4 {
  B4();
  InaccessibleDtor a[4]; // expected-note {{because field 'a' has an inaccessible destructor}}
};
B4 b4; // expected-error {{deleted function}}
union B5 {
  B5();
  // FIXME: Describe the anonymous union member better than ''.
  union { // expected-note {{because field '' has a deleted destructor}}
    DeletedDtor a; // expected-note {{because field 'a' has a deleted destructor}}
  };
};
B5 b5; // expected-error {{deleted function}}
union B6 {
  B6();
  union { // expected-note {{because field '' has a deleted destructor}}
    InaccessibleDtor a; // expected-note {{because field 'a' has an inaccessible destructor}}
  };
};
B6 b6; // expected-error {{deleted function}}

// -- any direct or virtual base class has a deleted or inaccessible destructor.
struct C1 : DeletedDtor { C1(); } c1; // expected-error {{deleted function}} expected-note {{base class 'DeletedDtor' has a deleted destructor}}
struct C2 : InaccessibleDtor { C2(); } c2; // expected-error {{deleted function}} expected-note {{base class 'InaccessibleDtor' has an inaccessible destructor}}
struct C3 : virtual DeletedDtor { C3(); } c3; // expected-error {{deleted function}} expected-note {{base class 'DeletedDtor' has a deleted destructor}}
struct C4 : virtual InaccessibleDtor { C4(); } c4; // expected-error {{deleted function}} expected-note {{base class 'InaccessibleDtor' has an inaccessible destructor}}

// -- for a virtual destructor, lookup of the non-array deallocation function
// results in an ambiguity or a function that is deleted or inaccessible.
class D1 {
  void operator delete(void*);
public:
  virtual ~D1() = default; // expected-note {{here}}
} d1; // ok
struct D2 : D1 { // expected-note 2{{virtual destructor requires an unambiguous, accessible 'operator delete'}} \
                 // expected-error {{deleted function '~D2' cannot override a non-deleted}}
  // implicitly-virtual destructor
} d2; // expected-error {{deleted function}}
struct D3 { // expected-note {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
  virtual ~D3() = default; // expected-note {{explicitly defaulted function was implicitly deleted here}}
  void operator delete(void*, double = 0.0);
  void operator delete(void*, char = 0);
} d3; // expected-error {{deleted function}}
struct D4 { // expected-note {{virtual destructor requires an unambiguous, accessible 'operator delete'}}
  virtual ~D4() = default; // expected-note {{implicitly deleted here}}
  void operator delete(void*) = delete;
} d4; // expected-error {{deleted function}}
