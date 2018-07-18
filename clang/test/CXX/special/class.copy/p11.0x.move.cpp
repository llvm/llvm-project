// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct Trivial {};
struct NonTrivial {
  NonTrivial(NonTrivial&&); // expected-note{{copy constructor is implicitly deleted}}
};
struct DeletedCopy {
  DeletedCopy(const DeletedCopy&) = delete;
};

// A defaulted move constructor for a class X is defined as deleted if X has:

// -- a variant member with a non-trivial corresponding constructor
union DeletedNTVariant {
  NonTrivial NT; // expected-note{{deleted because variant field 'NT' has a non-trivial move constructor}}
  DeletedNTVariant(DeletedNTVariant&&);
};
DeletedNTVariant::DeletedNTVariant(DeletedNTVariant&&) = default; // expected-error{{would delete}}

struct DeletedNTVariant2 {
  union {
    NonTrivial NT; // expected-note{{deleted because variant field 'NT' has a non-trivial move constructor}}
  };
  DeletedNTVariant2(DeletedNTVariant2&&);
};
DeletedNTVariant2::DeletedNTVariant2(DeletedNTVariant2&&) = default; // expected-error{{would delete}}

// Note, move constructor is not a candidate because it is deleted.
template<typename T> struct DeletedNTVariant3 { // expected-note 2{{default}} expected-note 2{{copy}}
  union {
    T NT;
  };
};
extern DeletedNTVariant3<NonTrivial> dntv3a(0); // expected-error {{no matching}}
extern DeletedNTVariant3<DeletedCopy> dntv3a(0); // expected-error {{no matching}}

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
  NoAccess NA; // expected-note{{deleted because field 'NA' has an inaccessible move constructor}}
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
  Ambiguity A; // expected-note{{deleted because field 'A' has multiple move constructors}}
  IsAmbiguous(IsAmbiguous&&); // expected-note{{copy constructor is implicitly deleted because 'IsAmbiguous' has a user-declared move constructor}}
};
IsAmbiguous::IsAmbiguous(IsAmbiguous&&) = default; // expected-error{{would delete}}

struct Deleted {
  // FIXME: This diagnostic is slightly wrong: the constructor we select to move
  // 'IA' is deleted, but we select the copy constructor (we ignore the move
  // constructor, because it was defaulted and deleted).
  IsAmbiguous IA; // expected-note{{deleted because field 'IA' has a deleted move constructor}}
  Deleted(Deleted&&);
};
Deleted::Deleted(Deleted&&) = default; // expected-error{{would delete}}

// It's implied (but not stated) that this should also happen if overload
// resolution fails.
struct ConstMember {
  const Trivial ct;
  ConstMember(ConstMember&&);
};
ConstMember::ConstMember(ConstMember&&) = default; // ok, calls copy ctor
struct ConstMoveOnlyMember {
  // FIXME: This diagnostic is slightly wrong: the constructor we select to move
  // 'cnt' is deleted, but we select the copy constructor, because the object is
  // const.
  const NonTrivial cnt; // expected-note{{deleted because field 'cnt' has a deleted move constructor}}
  ConstMoveOnlyMember(ConstMoveOnlyMember&&);
};
ConstMoveOnlyMember::ConstMoveOnlyMember(ConstMoveOnlyMember&&) = default; // expected-error{{would delete}}
struct VolatileMember {
  volatile Trivial vt; // expected-note{{deleted because field 'vt' has no move constructor}}
  VolatileMember(VolatileMember&&);
};
VolatileMember::VolatileMember(VolatileMember&&) = default; // expected-error{{would delete}}

// -- a direct or virtual base class B that cannot be moved because overload
//    resolution results in an ambiguity or a function that is deleted or
//    inaccessible
struct AmbiguousMoveBase : Ambiguity { // expected-note{{deleted because base class 'Ambiguity' has multiple move constructors}}
  AmbiguousMoveBase(AmbiguousMoveBase&&); // expected-note{{copy constructor is implicitly deleted}}
};
AmbiguousMoveBase::AmbiguousMoveBase(AmbiguousMoveBase&&) = default; // expected-error{{would delete}}

struct DeletedMoveBase : AmbiguousMoveBase { // expected-note{{deleted because base class 'AmbiguousMoveBase' has a deleted move constructor}}
  DeletedMoveBase(DeletedMoveBase&&);
};
DeletedMoveBase::DeletedMoveBase(DeletedMoveBase&&) = default; // expected-error{{would delete}}

struct InaccessibleMoveBase : NoAccess { // expected-note{{deleted because base class 'NoAccess' has an inaccessible move constructor}}
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
  NoAccessDtor NAD; // expected-note {{deleted because field 'NAD' has an inaccessible destructor}}
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
