// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Tests for implicit (non-)declaration of move constructor and
// assignment: p9, p11, p20, p23.

// This class, used as a member, allows to distinguish move from copy because
// move operations are no-throw, copy operations aren't.
struct ThrowingCopy {
  ThrowingCopy() noexcept;
  ThrowingCopy(ThrowingCopy &&) noexcept;
  ThrowingCopy(const ThrowingCopy &) noexcept(false);
  ThrowingCopy & operator =(ThrowingCopy &&) noexcept;
  ThrowingCopy & operator =(const ThrowingCopy &) noexcept(false);
};

struct HasCopyConstructor {
  ThrowingCopy tc;
  HasCopyConstructor() noexcept;
  HasCopyConstructor(const HasCopyConstructor &) noexcept(false);
};

struct HasCopyAssignment {
  ThrowingCopy tc;
  HasCopyAssignment() noexcept;
  HasCopyAssignment & operator =(const HasCopyAssignment &) noexcept(false);
};

struct HasMoveConstructor {
  ThrowingCopy tc;
  HasMoveConstructor() noexcept;
  HasMoveConstructor(HasMoveConstructor &&) noexcept; // expected-note {{copy assignment operator is implicitly deleted because 'HasMoveConstructor' has a user-declared move constructor}}
};

struct HasMoveAssignment { // expected-note {{implicit copy constructor}}
  ThrowingCopy tc;
  HasMoveAssignment() noexcept;
  HasMoveAssignment & operator =(HasMoveAssignment &&) noexcept;
};

struct HasDestructor {
  ThrowingCopy tc;
  HasDestructor() noexcept;
  ~HasDestructor() noexcept;
};

void test_basic_exclusion() {
  static_assert(!noexcept(HasCopyConstructor((HasCopyConstructor()))), "");
  HasCopyConstructor hcc;
  static_assert(!noexcept(hcc = HasCopyConstructor()), "");

  static_assert(!noexcept(HasCopyAssignment((HasCopyAssignment()))), "");
  HasCopyAssignment hca;
  static_assert(!noexcept(hca = HasCopyAssignment()), "");

  static_assert(noexcept(HasMoveConstructor((HasMoveConstructor()))), "");
  HasMoveConstructor hmc;
  hmc = HasMoveConstructor(); // expected-error {{selected implicitly-deleted copy assignment}}

  (HasMoveAssignment(HasMoveAssignment())); // expected-error {{uses deleted function}}
  HasMoveAssignment hma;
  static_assert(noexcept(hma = HasMoveAssignment()), "");

  static_assert(!noexcept(HasDestructor((HasDestructor()))), "");
  HasDestructor hd;
  static_assert(!noexcept(hd = HasDestructor()), "");
}

struct PrivateMove {
  PrivateMove() noexcept;
  PrivateMove(const PrivateMove &) noexcept(false);
  PrivateMove & operator =(const PrivateMove &) noexcept(false);
private:
  PrivateMove(PrivateMove &&) noexcept;
  PrivateMove & operator =(PrivateMove &&) noexcept;
};

struct InheritsPrivateMove : PrivateMove {};
struct ContainsPrivateMove {
  PrivateMove pm;
};

struct PrivateDestructor {
  PrivateDestructor() noexcept;
  PrivateDestructor(const PrivateDestructor &) noexcept(false);
  PrivateDestructor(PrivateDestructor &&) noexcept;
private:
  ~PrivateDestructor() noexcept;
};

struct InheritsPrivateDestructor : PrivateDestructor {}; // expected-note{{base class 'PrivateDestructor' has an inaccessible destructor}}
struct ContainsPrivateDestructor {
  PrivateDestructor pd; // expected-note{{field 'pd' has an inaccessible destructor}}
};

struct NonTrivialCopyOnly {
  NonTrivialCopyOnly() noexcept;
  NonTrivialCopyOnly(const NonTrivialCopyOnly &) noexcept(false);
  NonTrivialCopyOnly & operator =(const NonTrivialCopyOnly &) noexcept(false);
};

struct InheritsNonTrivialCopyOnly : NonTrivialCopyOnly {};
struct ContainsNonTrivialCopyOnly {
  NonTrivialCopyOnly ntco;
};

struct ContainsConst {
  const int i;
  ContainsConst() noexcept;
  ContainsConst & operator =(ContainsConst &); // expected-note {{not viable}}
};

struct ContainsRef {
  int &i;
  ContainsRef() noexcept;
  ContainsRef & operator =(ContainsRef &); // expected-note {{not viable}}
};

struct Base {
  Base & operator =(Base &);
};
struct DirectVirtualBase : virtual Base {}; // expected-note {{copy assignment operator) not viable}}
struct IndirectVirtualBase : DirectVirtualBase {}; // expected-note {{copy assignment operator) not viable}}

void test_deletion_exclusion() {
  // FIXME: How to test the union thing?

  static_assert(!noexcept(InheritsPrivateMove(InheritsPrivateMove())), "");
  static_assert(!noexcept(ContainsPrivateMove(ContainsPrivateMove())), "");
  InheritsPrivateMove ipm;
  static_assert(!noexcept(ipm = InheritsPrivateMove()), "");
  ContainsPrivateMove cpm;
  static_assert(!noexcept(cpm = ContainsPrivateMove()), "");

  (InheritsPrivateDestructor(InheritsPrivateDestructor())); // expected-error {{call to implicitly-deleted default constructor}}
  (ContainsPrivateDestructor(ContainsPrivateDestructor())); // expected-error {{call to implicitly-deleted default constructor}}

  static_assert(!noexcept(InheritsNonTrivialCopyOnly(InheritsNonTrivialCopyOnly())), "");
  static_assert(!noexcept(ContainsNonTrivialCopyOnly(ContainsNonTrivialCopyOnly())), "");
  InheritsNonTrivialCopyOnly intco;
  static_assert(!noexcept(intco = InheritsNonTrivialCopyOnly()), "");
  ContainsNonTrivialCopyOnly cntco;
  static_assert(!noexcept(cntco = ContainsNonTrivialCopyOnly()), "");

  ContainsConst cc;
  cc = ContainsConst(); // expected-error {{no viable}} 

  ContainsRef cr;
  cr = ContainsRef(); // expected-error {{no viable}} 

  DirectVirtualBase dvb;
  dvb = DirectVirtualBase(); // expected-error {{no viable}} 

  IndirectVirtualBase ivb;
  ivb = IndirectVirtualBase(); // expected-error {{no viable}} 
}

struct ContainsRValueRef {
  int&& ri;
  ContainsRValueRef() noexcept;
};

void test_contains_rref() {
  (ContainsRValueRef(ContainsRValueRef()));
}


namespace DR1402 {
  struct NonTrivialCopyCtor {
    NonTrivialCopyCtor(const NonTrivialCopyCtor &);
  };
  struct NonTrivialCopyAssign {
    NonTrivialCopyAssign &operator=(const NonTrivialCopyAssign &);
  };

  struct NonTrivialCopyCtorVBase : virtual NonTrivialCopyCtor {
    NonTrivialCopyCtorVBase(NonTrivialCopyCtorVBase &&);
    NonTrivialCopyCtorVBase &operator=(NonTrivialCopyCtorVBase &&) = default;
  };
  struct NonTrivialCopyAssignVBase : virtual NonTrivialCopyAssign {
    NonTrivialCopyAssignVBase(NonTrivialCopyAssignVBase &&);
    NonTrivialCopyAssignVBase &operator=(NonTrivialCopyAssignVBase &&) = default;
  };

  struct NonTrivialMoveAssign {
    NonTrivialMoveAssign(NonTrivialMoveAssign&&);
    NonTrivialMoveAssign &operator=(NonTrivialMoveAssign &&);
  };
  struct NonTrivialMoveAssignVBase : virtual NonTrivialMoveAssign {
    NonTrivialMoveAssignVBase(NonTrivialMoveAssignVBase &&);
    NonTrivialMoveAssignVBase &operator=(NonTrivialMoveAssignVBase &&) = default;
  };

  // A non-movable, non-trivially-copyable class type as a subobject inhibits
  // the declaration of a move operation.
  struct NoMove1 { NonTrivialCopyCtor ntcc; }; // expected-note 2{{'const DR1402::NoMove1 &'}}
  struct NoMove2 { NonTrivialCopyAssign ntcc; }; // expected-note 2{{'const DR1402::NoMove2 &'}}
  struct NoMove3 : NonTrivialCopyCtor {}; // expected-note 2{{'const DR1402::NoMove3 &'}}
  struct NoMove4 : NonTrivialCopyAssign {}; // expected-note 2{{'const DR1402::NoMove4 &'}}
  struct NoMove5 : virtual NonTrivialCopyCtor {}; // expected-note 2{{'const DR1402::NoMove5 &'}}
  struct NoMove6 : virtual NonTrivialCopyAssign {}; // expected-note 2{{'const DR1402::NoMove6 &'}}
  struct NoMove7 : NonTrivialCopyCtorVBase {}; // expected-note 2{{'const DR1402::NoMove7 &'}}
  struct NoMove8 : NonTrivialCopyAssignVBase {}; // expected-note 2{{'const DR1402::NoMove8 &'}}

  // A non-trivially-move-assignable virtual base class inhibits the declaration
  // of a move assignment (which might move-assign the base class multiple
  // times).
  struct NoMove9 : NonTrivialMoveAssign {};
  struct NoMove10 : virtual NonTrivialMoveAssign {}; // expected-note {{'const DR1402::NoMove10 &'}}
  struct NoMove11 : NonTrivialMoveAssignVBase {}; // expected-note {{'const DR1402::NoMove11 &'}}

  struct Test {
    friend NoMove1::NoMove1(NoMove1 &&); // expected-error {{no matching function}}
    friend NoMove2::NoMove2(NoMove2 &&); // expected-error {{no matching function}}
    friend NoMove3::NoMove3(NoMove3 &&); // expected-error {{no matching function}}
    friend NoMove4::NoMove4(NoMove4 &&); // expected-error {{no matching function}}
    friend NoMove5::NoMove5(NoMove5 &&); // expected-error {{no matching function}}
    friend NoMove6::NoMove6(NoMove6 &&); // expected-error {{no matching function}}
    friend NoMove7::NoMove7(NoMove7 &&); // expected-error {{no matching function}}
    friend NoMove8::NoMove8(NoMove8 &&); // expected-error {{no matching function}}
    friend NoMove9::NoMove9(NoMove9 &&);
    friend NoMove10::NoMove10(NoMove10 &&);
    friend NoMove11::NoMove11(NoMove11 &&);

    friend NoMove1 &NoMove1::operator=(NoMove1 &&); // expected-error {{no matching function}}
    friend NoMove2 &NoMove2::operator=(NoMove2 &&); // expected-error {{no matching function}}
    friend NoMove3 &NoMove3::operator=(NoMove3 &&); // expected-error {{no matching function}}
    friend NoMove4 &NoMove4::operator=(NoMove4 &&); // expected-error {{no matching function}}
    friend NoMove5 &NoMove5::operator=(NoMove5 &&); // expected-error {{no matching function}}
    friend NoMove6 &NoMove6::operator=(NoMove6 &&); // expected-error {{no matching function}}
    friend NoMove7 &NoMove7::operator=(NoMove7 &&); // expected-error {{no matching function}}
    friend NoMove8 &NoMove8::operator=(NoMove8 &&); // expected-error {{no matching function}}
    friend NoMove9 &NoMove9::operator=(NoMove9 &&);
    friend NoMove10 &NoMove10::operator=(NoMove10 &&); // expected-error {{no matching function}}
    friend NoMove11 &NoMove11::operator=(NoMove11 &&); // expected-error {{no matching function}}
  };
}

namespace PR12625 {
  struct X; // expected-note {{forward decl}}
  struct Y {
    X x; // expected-error {{incomplete}}
  } y = Y();
}
