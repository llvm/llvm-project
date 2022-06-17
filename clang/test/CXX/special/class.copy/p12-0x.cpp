// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-defaulted-function-deleted
// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-defaulted-function-deleted -fclang-abi-compat=14 -DCLANG_ABI_COMPAT=14

// expected-no-diagnostics

template<typename T, bool B> struct trivially_copyable_check {
  static_assert(B == __has_trivial_copy(T), "");
  static_assert(B == __is_trivially_constructible(T, T), "");
  static_assert(B == __is_trivially_constructible(T, const T &), "");
  static_assert(B == __is_trivially_constructible(T, T &&), "");
  typedef void type;
};
template<typename T> using trivially_copyable =
  typename trivially_copyable_check<T, true>::type;
template<typename T> using not_trivially_copyable =
  typename trivially_copyable_check<T, false>::type;

struct Trivial {};
using _ = trivially_copyable<Trivial>;

// A copy/move constructor for class X is trivial if it is not user-provided,
struct UserProvided {
  UserProvided(const UserProvided &);
};
using _ = not_trivially_copyable<UserProvided>;

// its declared parameter type is the same as if it had been implicitly
// declared,
struct NonConstCopy {
  NonConstCopy(NonConstCopy &) = default;
};
#if defined(CLANG_ABI_COMPAT) && CLANG_ABI_COMPAT <= 14
// Up until (and including) Clang 14, non-const copy constructors were not trivial because of dr2171
using _ = not_trivially_copyable<NonConstCopy>;
#else
// In the latest Clang version, all defaulted constructors are trivial, even if non-const, because
// dr2171 is fixed.
static_assert(__has_trivial_copy(NonConstCopy), "");
static_assert(__is_trivially_constructible(NonConstCopy, NonConstCopy &), "");
static_assert(!__is_trivially_constructible(NonConstCopy, NonConstCopy), "");
static_assert(!__is_trivially_constructible(NonConstCopy, const NonConstCopy &), "");
static_assert(!__is_trivially_constructible(NonConstCopy, NonConstCopy &&), "");

struct DefaultedSpecialMembers {
  DefaultedSpecialMembers(const DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers(DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers(DefaultedSpecialMembers &&) = default;
};
using _ = trivially_copyable<DefaultedSpecialMembers>;
#endif

// class X has no virtual functions
struct VFn {
  virtual void f();
};
using _ = not_trivially_copyable<VFn>;

// and no virtual base classes
struct VBase : virtual Trivial {};
using _ = not_trivially_copyable<VBase>;

// and the constructor selected to copy/move each [direct subobject] is trivial
struct TemplateCtor {
  template<typename T> TemplateCtor(T &);
};
using _ = trivially_copyable<TemplateCtor>;
struct TemplateCtorMember {
  TemplateCtor tc;
};
using _ = trivially_copyable<TemplateCtorMember>;

// We can select a non-trivial copy ctor even if there is a trivial one.
struct MutableTemplateCtorMember {
  mutable TemplateCtor mtc;
};
static_assert(!__is_trivially_constructible(MutableTemplateCtorMember, const MutableTemplateCtorMember &), "");
static_assert(__is_trivially_constructible(MutableTemplateCtorMember, MutableTemplateCtorMember &&), "");
struct MutableTemplateCtorMember2 {
  MutableTemplateCtorMember2(const MutableTemplateCtorMember2 &) = default;
  MutableTemplateCtorMember2(MutableTemplateCtorMember2 &&) = default;
  mutable TemplateCtor mtc;
};
static_assert(!__is_trivially_constructible(MutableTemplateCtorMember2, const MutableTemplateCtorMember2 &), "");
static_assert(__is_trivially_constructible(MutableTemplateCtorMember2, MutableTemplateCtorMember2 &&), "");

// Both trivial and non-trivial special members.
struct TNT {
  TNT(const TNT &) = default; // trivial
  TNT(TNT &); // non-trivial

  TNT(TNT &&) = default; // trivial
  TNT(const TNT &&); // non-trivial
};

static_assert(!__has_trivial_copy(TNT), "lie deliberately for gcc compatibility");
static_assert(__is_trivially_constructible(TNT, TNT), "");
static_assert(!__is_trivially_constructible(TNT, TNT &), "");
static_assert(__is_trivially_constructible(TNT, const TNT &), "");
static_assert(!__is_trivially_constructible(TNT, volatile TNT &), "");
static_assert(__is_trivially_constructible(TNT, TNT &&), "");
static_assert(!__is_trivially_constructible(TNT, const TNT &&), "");
static_assert(!__is_trivially_constructible(TNT, volatile TNT &&), "");

// This has only trivial special members.
struct DerivedFromTNT : TNT {};

static_assert(__has_trivial_copy(DerivedFromTNT), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT &), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, const DerivedFromTNT &), "");
static_assert(!__is_trivially_constructible(DerivedFromTNT, volatile DerivedFromTNT &), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, DerivedFromTNT &&), "");
static_assert(__is_trivially_constructible(DerivedFromTNT, const DerivedFromTNT &&), "");
static_assert(!__is_trivially_constructible(DerivedFromTNT, volatile DerivedFromTNT &&), "");

// This has only trivial special members.
struct TNTMember {
  TNT tnt;
};

static_assert(__has_trivial_copy(TNTMember), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember &), "");
static_assert(__is_trivially_constructible(TNTMember, const TNTMember &), "");
static_assert(!__is_trivially_constructible(TNTMember, volatile TNTMember &), "");
static_assert(__is_trivially_constructible(TNTMember, TNTMember &&), "");
static_assert(__is_trivially_constructible(TNTMember, const TNTMember &&), "");
static_assert(!__is_trivially_constructible(TNTMember, volatile TNTMember &&), "");

struct NCCTNT : NonConstCopy, TNT {};

static_assert(!__has_trivial_copy(NCCTNT), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, const NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, volatile NCCTNT &), "");
static_assert(!__is_trivially_constructible(NCCTNT, NCCTNT &&), "");
static_assert(!__is_trivially_constructible(NCCTNT, const NCCTNT &&), "");
static_assert(!__is_trivially_constructible(NCCTNT, volatile NCCTNT &&), "");

struct TemplateCtorNoMove {
  TemplateCtorNoMove(const TemplateCtorNoMove &) = default;
  template<typename T> TemplateCtorNoMove(T &&);
};
static_assert(__is_trivially_constructible(TemplateCtorNoMove, const TemplateCtorNoMove &), "");
static_assert(!__is_trivially_constructible(TemplateCtorNoMove, TemplateCtorNoMove &&), "");

struct UseTemplateCtorNoMove {
  TemplateCtorNoMove tcnm;
};
static_assert(__is_trivially_constructible(UseTemplateCtorNoMove, const UseTemplateCtorNoMove &), "");
static_assert(!__is_trivially_constructible(UseTemplateCtorNoMove, UseTemplateCtorNoMove &&), "");

struct TemplateCtorNoMoveSFINAE {
  TemplateCtorNoMoveSFINAE(const TemplateCtorNoMoveSFINAE &) = default;
  template<typename T, typename U = typename T::error> TemplateCtorNoMoveSFINAE(T &&);
};
static_assert(__is_trivially_constructible(TemplateCtorNoMoveSFINAE, const TemplateCtorNoMoveSFINAE &), "");
static_assert(__is_trivially_constructible(TemplateCtorNoMoveSFINAE, TemplateCtorNoMoveSFINAE &&), "");

struct UseTemplateCtorNoMoveSFINAE {
  TemplateCtorNoMoveSFINAE tcnm;
};
static_assert(__is_trivially_constructible(UseTemplateCtorNoMoveSFINAE, const UseTemplateCtorNoMoveSFINAE &), "");
static_assert(__is_trivially_constructible(UseTemplateCtorNoMoveSFINAE, UseTemplateCtorNoMoveSFINAE &&), "");

namespace TrivialityDependsOnImplicitDeletion {
  struct PrivateMove {
    PrivateMove(const PrivateMove &) = default;
  private:
    PrivateMove(PrivateMove &&);
    friend class Access;
  };
  static_assert(__is_trivially_constructible(PrivateMove, const PrivateMove &), "");
  static_assert(!__is_trivially_constructible(PrivateMove, PrivateMove &&), "");

  struct NoAccess {
    PrivateMove pm;
    // NoAccess's move is deleted, so moves of it use PrivateMove's copy ctor,
    // which is trivial.
  };
  static_assert(__is_trivially_constructible(NoAccess, const NoAccess &), "");
  static_assert(__is_trivially_constructible(NoAccess, NoAccess &&), "");
  struct TopNoAccess : NoAccess {};
  static_assert(__is_trivially_constructible(TopNoAccess, const TopNoAccess &), "");
  static_assert(__is_trivially_constructible(TopNoAccess, TopNoAccess &&), "");

  struct Access {
    PrivateMove pm;
    // NoAccess's move would *not* be deleted, so is *not* suppressed,
    // so moves of it use PrivateMove's move ctor, which is not trivial.
  };
  static_assert(__is_trivially_constructible(Access, const Access &), "");
  static_assert(!__is_trivially_constructible(Access, Access &&), "");
  struct TopAccess : Access {};
  static_assert(__is_trivially_constructible(TopAccess, const TopAccess &), "");
  static_assert(!__is_trivially_constructible(TopAccess, TopAccess &&), "");
}

namespace TrivialityDependsOnDestructor {
  class HasInaccessibleDestructor { ~HasInaccessibleDestructor() = default; };
  struct HasImplicitlyDeletedDestructor : HasInaccessibleDestructor {};
  struct HasImplicitlyDeletedCopyCtor : HasImplicitlyDeletedDestructor {
    HasImplicitlyDeletedCopyCtor() = default;
    template<typename T> HasImplicitlyDeletedCopyCtor(T &&);
    // Copy ctor is deleted but trivial.
    // Move ctor is suppressed.
    HasImplicitlyDeletedCopyCtor(const HasImplicitlyDeletedCopyCtor&) = default;
    HasImplicitlyDeletedCopyCtor(HasImplicitlyDeletedCopyCtor&&) = default;
  };
  struct Test : HasImplicitlyDeletedCopyCtor {
    Test(const Test&) = default;
    Test(Test&&) = default;
  };
  // Implicit copy ctor calls deleted trivial copy ctor.
  static_assert(__has_trivial_copy(Test), "");
  // This is false because the destructor is deleted.
  static_assert(!__is_trivially_constructible(Test, const Test &), "");
  // Implicit move ctor calls template ctor.
  static_assert(!__is_trivially_constructible(Test, Test &&), "");

  struct HasAccessibleDestructor { ~HasAccessibleDestructor() = default; };
  struct HasImplicitlyDefaultedDestructor : HasAccessibleDestructor {};
  struct HasImplicitlyDefaultedCopyCtor : HasImplicitlyDefaultedDestructor {
    template<typename T> HasImplicitlyDefaultedCopyCtor(T &&);
    // Copy ctor is trivial.
    // Move ctor is trivial.
  };
  struct Test2 : HasImplicitlyDefaultedCopyCtor {};
  // Implicit copy ctor calls trivial copy ctor.
  static_assert(__has_trivial_copy(Test2), "");
  static_assert(__is_trivially_constructible(Test2, const Test2 &), "");
  // Implicit move ctor calls trivial move ctor.
  static_assert(__is_trivially_constructible(Test2, Test2 &&), "");
}
