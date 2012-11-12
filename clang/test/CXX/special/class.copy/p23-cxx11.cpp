// RUN: %clang_cc1 -verify %s -std=c++11

template<typename T> struct CopyAssign {
  static T t;
  void test() {
    t = t; // expected-error +{{deleted}}
  }
};
template<typename T> struct MoveAssign {
  static T t;
  void test() {
    t = static_cast<T&&>(t); // expected-error +{{deleted}}
  }
};

struct NonTrivialCopyAssign {
  NonTrivialCopyAssign &operator=(const NonTrivialCopyAssign &);
};
struct NonTrivialMoveAssign {
  NonTrivialMoveAssign &operator=(NonTrivialMoveAssign &&);
};
struct AmbiguousCopyAssign {
  AmbiguousCopyAssign &operator=(const AmbiguousCopyAssign &);
  AmbiguousCopyAssign &operator=(volatile AmbiguousCopyAssign &);
};
struct AmbiguousMoveAssign {
  AmbiguousMoveAssign &operator=(const AmbiguousMoveAssign &&);
  AmbiguousMoveAssign &operator=(volatile AmbiguousMoveAssign &&);
};
struct DeletedCopyAssign {
  DeletedCopyAssign &operator=(const DeletedCopyAssign &) = delete; // expected-note 2{{deleted}}
};
struct DeletedMoveAssign {
  DeletedMoveAssign &operator=(DeletedMoveAssign &&) = delete; // expected-note 2{{deleted}}
};
class InaccessibleCopyAssign {
  InaccessibleCopyAssign &operator=(const InaccessibleCopyAssign &);
};
class InaccessibleMoveAssign {
  InaccessibleMoveAssign &operator=(InaccessibleMoveAssign &&);
};

// A defaulted copy/move assignment operator for class X is defined as deleted
// if X has:

//   -- a variant member with a non-trivial corresponding assignment operator
//      and X is a union-like class
struct A1 {
  union {
    NonTrivialCopyAssign x; // expected-note {{variant field 'x' has a non-trivial copy assign}}
  };
};
template struct CopyAssign<A1>; // expected-note {{here}}

struct A2 {
  A2 &operator=(A2 &&) = default; // expected-note {{here}}
  union {
    NonTrivialMoveAssign x; // expected-note {{variant field 'x' has a non-trivial move assign}}
  };
};
template struct MoveAssign<A2>; // expected-note {{here}}

//   -- a non-static const data member of (array of) non-class type
struct B1 {
  const int a; // expected-note 2{{field 'a' is of const-qualified type}}
};
struct B2 {
  const void *const a[3][9][2]; // expected-note 2{{field 'a' is of const-qualified type 'const void *const [3][9][2]'}}
};
struct B3 {
  const void *a[3];
};
template struct CopyAssign<B1>; // expected-note {{here}}
template struct MoveAssign<B1>; // expected-note {{here}}
template struct CopyAssign<B2>; // expected-note {{here}}
template struct MoveAssign<B2>; // expected-note {{here}}
template struct CopyAssign<B3>;
template struct MoveAssign<B3>;

//   -- a non-static data member of reference type
struct C1 {
  int &a; // expected-note 2{{field 'a' is of reference type 'int &'}}
};
template struct CopyAssign<C1>; // expected-note {{here}}
template struct MoveAssign<C1>; // expected-note {{here}}

//   -- a non-static data member of class type M that cannot be copied/moved
struct D1 {
  AmbiguousCopyAssign a; // expected-note {{field 'a' has multiple copy}}
};
struct D2 {
  D2 &operator=(D2 &&) = default; // expected-note {{here}}
  AmbiguousMoveAssign a; // expected-note {{field 'a' has multiple move}}
};
struct D3 {
  DeletedCopyAssign a; // expected-note {{field 'a' has a deleted copy}}
};
struct D4 {
  D4 &operator=(D4 &&) = default; // expected-note {{here}}
  DeletedMoveAssign a; // expected-note {{field 'a' has a deleted move}}
};
struct D5 {
  InaccessibleCopyAssign a; // expected-note {{field 'a' has an inaccessible copy}}
};
struct D6 {
  D6 &operator=(D6 &&) = default; // expected-note {{here}}
  InaccessibleMoveAssign a; // expected-note {{field 'a' has an inaccessible move}}
};
template struct CopyAssign<D1>; // expected-note {{here}}
template struct MoveAssign<D2>; // expected-note {{here}}
template struct CopyAssign<D3>; // expected-note {{here}}
template struct MoveAssign<D4>; // expected-note {{here}}
template struct CopyAssign<D5>; // expected-note {{here}}
template struct MoveAssign<D6>; // expected-note {{here}}

//   -- a direct or virtual base that cannot be copied/moved
struct E1 : AmbiguousCopyAssign {}; // expected-note {{base class 'AmbiguousCopyAssign' has multiple copy}}
struct E2 : AmbiguousMoveAssign { // expected-note {{base class 'AmbiguousMoveAssign' has multiple move}}
  E2 &operator=(E2 &&) = default; // expected-note {{here}}
};
struct E3 : DeletedCopyAssign {}; // expected-note {{base class 'DeletedCopyAssign' has a deleted copy}}
struct E4 : DeletedMoveAssign { // expected-note {{base class 'DeletedMoveAssign' has a deleted move}}
  E4 &operator=(E4 &&) = default; // expected-note {{here}}
};
struct E5 : InaccessibleCopyAssign {}; // expected-note {{base class 'InaccessibleCopyAssign' has an inaccessible copy}}
struct E6 : InaccessibleMoveAssign { // expected-note {{base class 'InaccessibleMoveAssign' has an inaccessible move}}
  E6 &operator=(E6 &&) = default; // expected-note {{here}}
};
template struct CopyAssign<E1>; // expected-note {{here}}
template struct MoveAssign<E2>; // expected-note {{here}}
template struct CopyAssign<E3>; // expected-note {{here}}
template struct MoveAssign<E4>; // expected-note {{here}}
template struct CopyAssign<E5>; // expected-note {{here}}
template struct MoveAssign<E6>; // expected-note {{here}}

namespace PR13381 {
  struct S {
    S &operator=(const S&);
    S &operator=(const volatile S&) = delete; // expected-note{{deleted here}}
  };
  struct T {
    volatile S s; // expected-note{{field 's' has a deleted copy assignment}}
  };
  void g() {
    T t;
    t = T(); // expected-error{{implicitly-deleted copy assignment}}
  }
}
