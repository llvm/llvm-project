// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct NonTrivial {
  NonTrivial(NonTrivial&&);
};

// A defaulted move constructor for a class X is defined as deleted if X has:

// -- a variant member with a non-trivial corresponding constructor
union DeletedNTVariant {
  NonTrivial NT;
  DeletedNTVariant(DeletedNTVariant&&);
};
DeletedNTVariant::DeletedNTVariant(DeletedNTVariant&&) = default; // expected-error{{would delete}}

struct DeletedNTVariant2 {
  union {
    NonTrivial NT;
  };
  DeletedNTVariant2(DeletedNTVariant2&&);
};
DeletedNTVariant2::DeletedNTVariant2(DeletedNTVariant2&&) = default; // expected-error{{would delete}}

// -- a non-static data member of class type M (or array thereof) that cannot be
//    copied because overload resolution results in an ambiguity or a function
//    that is deleted or inaccessible
struct NoAccess {
  NoAccess() = default;
private:
  NoAccess(NoAccess&&);

  friend struct HasAccess;
};

struct HasNoAccess {
  NoAccess NA;
  HasNoAccess(HasNoAccess&&);
};
HasNoAccess::HasNoAccess(HasNoAccess&&) = default; // expected-error{{would delete}}

struct HasAccess {
  NoAccess NA;
  HasAccess(HasAccess&&);
};
HasAccess::HasAccess(HasAccess&&) = default;

struct Ambiguity {
  Ambiguity(const Ambiguity&&);
  Ambiguity(volatile Ambiguity&&);
};

struct IsAmbiguous {
  Ambiguity A;
  IsAmbiguous(IsAmbiguous&&);
};
IsAmbiguous::IsAmbiguous(IsAmbiguous&&) = default; // expected-error{{would delete}}

struct Deleted {
  IsAmbiguous IA;
  Deleted(Deleted&&);
};
Deleted::Deleted(Deleted&&) = default; // expected-error{{would delete}}

// -- a direct or virtual base class B that cannot be moved because overload
//    resolution results in an ambiguity or a function that is deleted or
//    inaccessible
struct AmbiguousMoveBase : Ambiguity {
  AmbiguousMoveBase(AmbiguousMoveBase&&);
};
AmbiguousMoveBase::AmbiguousMoveBase(AmbiguousMoveBase&&) = default; // expected-error{{would delete}}

struct DeletedMoveBase : AmbiguousMoveBase {
  DeletedMoveBase(DeletedMoveBase&&);
};
DeletedMoveBase::DeletedMoveBase(DeletedMoveBase&&) = default; // expected-error{{would delete}}

struct InaccessibleMoveBase : NoAccess {
  InaccessibleMoveBase(InaccessibleMoveBase&&);
};
InaccessibleMoveBase::InaccessibleMoveBase(InaccessibleMoveBase&&) = default; // expected-error{{would delete}}

// -- any direct or virtual base class or non-static data member of a type with
//    a destructor that is deleted or inaccessible
struct NoAccessDtor {
  NoAccessDtor(NoAccessDtor&&); // expected-note{{copy constructor is implicitly deleted because 'NoAccessDtor' has a user-declared move constructor}}
private:
  ~NoAccessDtor();
  friend struct HasAccessDtor;
};

struct HasNoAccessDtor {
  NoAccessDtor NAD;
  HasNoAccessDtor(HasNoAccessDtor&&);
};
HasNoAccessDtor::HasNoAccessDtor(HasNoAccessDtor&&) = default; // expected-error{{would delete}}

struct HasAccessDtor {
  NoAccessDtor NAD;
  HasAccessDtor(HasAccessDtor&&);
};
HasAccessDtor::HasAccessDtor(HasAccessDtor&&) = default;

struct HasNoAccessDtorBase : NoAccessDtor { // expected-note{{copy constructor of 'HasNoAccessDtorBase' is implicitly deleted because base class 'NoAccessDtor' has a deleted copy constructor}}
};
extern HasNoAccessDtorBase HNADBa;
HasNoAccessDtorBase HNADBb(HNADBa); // expected-error{{implicitly-deleted copy constructor}}

// The restriction on rvalue reference members applies to only the copy
// constructor.
struct RValue {
  int &&ri = 1;
  RValue(RValue&&);
};
RValue::RValue(RValue&&) = default;

// -- a non-static data member or direct or virtual base class with a type that
//    does not have a move constructor and is not trivially copyable
struct CopyOnly {
  CopyOnly(const CopyOnly&);
};

struct NonMove {
  CopyOnly CO;
  NonMove(NonMove&&);
};
NonMove::NonMove(NonMove&&) = default; // ok under DR1402

struct Moveable {
  Moveable();
  Moveable(Moveable&&);
};

struct HasMove {
  Moveable M;
  HasMove(HasMove&&);
};
HasMove::HasMove(HasMove&&) = default;

namespace DR1402 {
  struct member {
    member();
    member(const member&);
    member& operator=(const member&);
    ~member();
  };

  struct A {
    member m_;

    A() = default;
    A(const A&) = default;
    A& operator=(const A&) = default;
    A(A&&) = default;
    A& operator=(A&&) = default;
    ~A() = default;
  };

  // ok, A's explicitly-defaulted move operations copy m_.
  void f() {
    A a, b(a), c(static_cast<A&&>(a));
    a = b;
    b = static_cast<A&&>(c);
  }
}
