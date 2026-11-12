// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wno-inaccessible-base %s

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
  hmc = HasMoveConstructor(); // expected-error {{object of type 'HasMoveConstructor' cannot be assigned because its copy assignment operator is implicitly deleted}}

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

  // DR1402: A non-movable, non-trivially-copyable class type as a subobject no
  // longer inhibits the declaration of a move operation.
  struct NoMove1 { NonTrivialCopyCtor ntcc; };
  struct NoMove2 { NonTrivialCopyAssign ntcc; };
  struct NoMove3 : NonTrivialCopyCtor {};
  struct NoMove4 : NonTrivialCopyAssign {};
  struct NoMove5 : virtual NonTrivialCopyCtor {};
  struct NoMove6 : virtual NonTrivialCopyAssign {};
  struct NoMove7 : NonTrivialCopyCtorVBase {};
  struct NoMove8 : NonTrivialCopyAssignVBase {};

  // DR1402: A non-trivially-move-assignable virtual base class no longer
  // inhibits the declaration of a move assignment (even though it might
  // move-assign the base class multiple times).
  struct NoMove9 : NonTrivialMoveAssign {};
  struct NoMove10 : virtual NonTrivialMoveAssign {};
  struct NoMove11 : NonTrivialMoveAssignVBase {};

  template<typename T> void test(T t) {
    (void)T(static_cast<T&&>(t)); // ok
    t = static_cast<T&&>(t); // ok
  }
  template void test(NoMove1);
  template void test(NoMove2);
  template void test(NoMove3);
  template void test(NoMove4);
  template void test(NoMove5);
  template void test(NoMove6);
  template void test(NoMove7);
  template void test(NoMove8);
  template void test(NoMove9);
  template void test(NoMove10);
  template void test(NoMove11);

  struct CopyOnly {
    CopyOnly(const CopyOnly&);
    CopyOnly &operator=(const CopyOnly&);
  };
  struct MoveOnly {
    MoveOnly(MoveOnly&&); // expected-note {{user-declared move}}
    MoveOnly &operator=(MoveOnly&&);
  };
  template void test(CopyOnly); // ok, copies
  template void test(MoveOnly); // ok, moves
  struct CopyAndMove { // expected-note {{implicitly deleted}}
    CopyOnly co;
    MoveOnly mo; // expected-note {{deleted copy}}
  };
  template void test(CopyAndMove); // ok, copies co, moves mo
  void test2(CopyAndMove cm) {
    (void)CopyAndMove(cm); // expected-error {{deleted}}
    cm = cm; // expected-error {{deleted}}
  }

  namespace VbaseMove {
    struct A {};
    struct B { B &operator=(B&&); };
    struct C { C &operator=(const C&); };
    struct D { B b; };

    template<typename T, unsigned I, bool NonTrivialMove = false>
    struct E : virtual T {};

    template<typename T, unsigned I>
    struct E<T, I, true> : virtual T { E &operator=(E&&); };

    template<typename T>
    struct F :
      E<T, 0>, // expected-note-re 2{{'{{[BD]}}' is a virtual base class of base class 'E<}}
      E<T, 1> {}; // expected-note-re 2{{'{{[BD]}}' is a virtual base class of base class 'E<}}

    template<typename T>
    struct G : E<T, 0, true>, E<T, 0> {};

    template<typename T>
    struct H : E<T, 0, true>, E<T, 1, true> {};

    template<typename T>
    struct I : E<T, 0>, T {};

    template<typename T>
    struct J :
      E<T, 0>, // expected-note-re 2{{'{{[BD]}}' is a virtual base class of base class 'E<}}
      virtual T {}; // expected-note-re 2{{virtual base class '{{[BD]}}' declared here}}

    template<typename T> void move(T t) { t = static_cast<T&&>(t); }
    // expected-warning-re@-1 4{{defaulted move assignment operator of {{.*}} will move assign virtual base class '{{[BD]}}' multiple times}}
    template void move(F<A>);
    template void move(F<B>); // expected-note {{in instantiation of}}
    template void move(F<C>);
    template void move(F<D>); // expected-note {{in instantiation of}}
    template void move(G<A>);
    template void move(G<B>);
    template void move(G<C>);
    template void move(G<D>);
    template void move(H<A>);
    template void move(H<B>);
    template void move(H<C>);
    template void move(H<D>);
    template void move(I<A>);
    template void move(I<B>);
    template void move(I<C>);
    template void move(I<D>);
    template void move(J<A>);
    template void move(J<B>); // expected-note {{in instantiation of}}
    template void move(J<C>);
    template void move(J<D>); // expected-note {{in instantiation of}}
  }
}

namespace PR12625 {
  struct X; // expected-note {{forward decl}}
  struct Y {
    X x; // expected-error {{incomplete}}
  } y = Y();
}
