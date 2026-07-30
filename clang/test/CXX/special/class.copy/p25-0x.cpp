// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-deprecated-builtins
// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-deprecated-builtins -fclang-abi-compat=14 -DCLANG_ABI_COMPAT=14
// RUN: %clang_cc1 -fsyntax-only -std=c++2b -DDEDUCING_THIS -Wno-deprecated-builtins  %s -verify

// expected-no-diagnostics

#if DEDUCING_THIS
#define EXPLICIT_PARAMETER(...) this __VA_ARGS__,
#else
#define EXPLICIT_PARAMETER(Param)
#endif


template<typename T, bool B> struct trivially_assignable_check {
  static_assert(B == __has_trivial_assign(T), "");
  static_assert(B == __is_trivially_assignable(T&, T), "");
  static_assert(B == __is_trivially_assignable(T&, const T &), "");
  static_assert(B == __is_trivially_assignable(T&, T &&), "");
  static_assert(B == __is_trivially_assignable(T&&, T), "");
  static_assert(B == __is_trivially_assignable(T&&, const T &), "");
  static_assert(B == __is_trivially_assignable(T&&, T &&), "");
  typedef void type;
};
template<typename T> using trivially_assignable =
  typename trivially_assignable_check<T, true>::type;
template<typename T> using not_trivially_assignable =
  typename trivially_assignable_check<T, false>::type;

struct Trivial {};
using _ = trivially_assignable<Trivial>;

// A copy/move assignment operator for class X is trivial if it is not user-provided,
struct UserProvided {
  UserProvided &operator=(EXPLICIT_PARAMETER(UserProvided&)
                          const UserProvided &);
};
using _ = not_trivially_assignable<UserProvided>;

// its declared parameter type is the same as if it had been implicitly
// declared,
struct NonConstCopy {
  NonConstCopy &operator=(EXPLICIT_PARAMETER(NonConstCopy&) NonConstCopy &) = default;
#if DEDUCING_THIS
  NonConstCopy &operator=(EXPLICIT_PARAMETER(NonConstCopy&&) NonConstCopy &) = default;
#endif
};
#if defined(CLANG_ABI_COMPAT) && CLANG_ABI_COMPAT <= 14
// Up until (and including) Clang 14, non-const copy assignment operators were not trivial because
// of dr2171
using _ = not_trivially_assignable<NonConstCopy>;
#else
// In the latest Clang version, all defaulted assignment operators are trivial, even if non-const,
// because dr2171 is fixed
static_assert(__has_trivial_assign(NonConstCopy), "");
static_assert(__is_trivially_assignable(NonConstCopy &, NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, const NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, NonConstCopy), "");
static_assert(!__is_trivially_assignable(NonConstCopy &, NonConstCopy &&), "");
static_assert(__is_trivially_assignable(NonConstCopy &&, NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, const NonConstCopy &), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, NonConstCopy), "");
static_assert(!__is_trivially_assignable(NonConstCopy &&, NonConstCopy &&), "");

struct DefaultedSpecialMembers {
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&)
                                     const DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&)
                                     DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&)
                                     DefaultedSpecialMembers &&) = default;
#if DEDUCING_THIS
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&&)
                                     const DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&&)
                                     DefaultedSpecialMembers &) = default;
  DefaultedSpecialMembers &operator=(EXPLICIT_PARAMETER(DefaultedSpecialMembers&&)
                                     DefaultedSpecialMembers &&) = default;
#endif
};
using _ = trivially_assignable<DefaultedSpecialMembers>;
#endif

// class X has no virtual functions
struct VFn {
  virtual void f();
};
using _ = not_trivially_assignable<VFn>;

// and no virtual base classes
struct VBase : virtual Trivial {};
using _ = not_trivially_assignable<VBase>;

// and the assignment operator selected to copy/move each [direct subobject] is trivial
struct TemplateCtor {
  template<typename T> TemplateCtor operator=(T &);
};
using _ = trivially_assignable<TemplateCtor>;
struct TemplateCtorMember {
  TemplateCtor tc;
};
using _ = trivially_assignable<TemplateCtorMember>;
struct MutableTemplateCtorMember {
  mutable TemplateCtor mtc;
};
static_assert(!__is_trivially_assignable(MutableTemplateCtorMember, const MutableTemplateCtorMember &), "");
static_assert(__is_trivially_assignable(MutableTemplateCtorMember, MutableTemplateCtorMember &&), "");

// Both trivial and non-trivial special members.
struct TNT {
  TNT &operator=(EXPLICIT_PARAMETER(TNT&) const TNT &) = default; // trivial
  TNT &operator=(EXPLICIT_PARAMETER(TNT&) TNT &); // non-trivial
  TNT &operator=(EXPLICIT_PARAMETER(TNT&) TNT &&) = default; // trivial
  TNT &operator=(EXPLICIT_PARAMETER(TNT&) const TNT &&); // non-trivial
#if DEDUCING_THIS
  TNT &operator=(this TNT&&, const TNT &) = default; // trivial
  TNT &operator=(this TNT&&, TNT &); // non-trivial
  TNT &operator=(this TNT&&, TNT &&) = default; // trivial
  TNT &operator=(this TNT&&, const TNT &&); // non-trivial
#endif
};

static_assert(!__has_trivial_assign(TNT), "lie deliberately for gcc compatibility");
static_assert(__is_trivially_assignable(TNT, TNT), "");
static_assert(!__is_trivially_assignable(TNT, TNT &), "");
static_assert(__is_trivially_assignable(TNT, const TNT &), "");
static_assert(!__is_trivially_assignable(TNT, volatile TNT &), "");
static_assert(__is_trivially_assignable(TNT, TNT &&), "");
static_assert(!__is_trivially_assignable(TNT, const TNT &&), "");
static_assert(!__is_trivially_assignable(TNT, volatile TNT &&), "");

// This has only trivial special members.
struct DerivedFromTNT : TNT {};

static_assert(__has_trivial_assign(DerivedFromTNT), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT &), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, const DerivedFromTNT &), "");
static_assert(!__is_trivially_assignable(DerivedFromTNT, volatile DerivedFromTNT &), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, DerivedFromTNT &&), "");
static_assert(__is_trivially_assignable(DerivedFromTNT, const DerivedFromTNT &&), "");
static_assert(!__is_trivially_assignable(DerivedFromTNT, volatile DerivedFromTNT &&), "");

// This has only trivial special members.
struct TNTMember {
  TNT tnt;
};

static_assert(__has_trivial_assign(TNTMember), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember &), "");
static_assert(__is_trivially_assignable(TNTMember, const TNTMember &), "");
static_assert(!__is_trivially_assignable(TNTMember, volatile TNTMember &), "");
static_assert(__is_trivially_assignable(TNTMember, TNTMember &&), "");
static_assert(__is_trivially_assignable(TNTMember, const TNTMember &&), "");
static_assert(!__is_trivially_assignable(TNTMember, volatile TNTMember &&), "");

struct NCCTNT : NonConstCopy, TNT {};

static_assert(!__has_trivial_assign(NCCTNT), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, const NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, volatile NCCTNT &), "");
static_assert(!__is_trivially_assignable(NCCTNT, NCCTNT &&), "");
static_assert(!__is_trivially_assignable(NCCTNT, const NCCTNT &&), "");
static_assert(!__is_trivially_assignable(NCCTNT, volatile NCCTNT &&), "");

struct MultipleTrivial {
  // All four of these are trivial.
  MultipleTrivial &operator=(const MultipleTrivial &) & = default;
  MultipleTrivial &operator=(const MultipleTrivial &) && = default;
  MultipleTrivial &operator=(MultipleTrivial &&) & = default;
  MultipleTrivial &operator=(MultipleTrivial &&) && = default;
};

using _ = trivially_assignable<MultipleTrivial>;

struct RefQualifier {
  RefQualifier &operator=(const RefQualifier &) & = default;
  RefQualifier &operator=(const RefQualifier &) &&;
  RefQualifier &operator=(RefQualifier &&) &;
  RefQualifier &operator=(RefQualifier &&) && = default;
};
struct DerivedFromRefQualifier : RefQualifier {
  // Both of these call the trivial copy operation.
  DerivedFromRefQualifier &operator=(const DerivedFromRefQualifier &) & = default;
  DerivedFromRefQualifier &operator=(const DerivedFromRefQualifier &) && = default;
  // Both of these call the non-trivial move operation.
  DerivedFromRefQualifier &operator=(DerivedFromRefQualifier &&) & = default;
  DerivedFromRefQualifier &operator=(DerivedFromRefQualifier &&) && = default;
};
static_assert(__is_trivially_assignable(DerivedFromRefQualifier&, const DerivedFromRefQualifier&), "");
static_assert(__is_trivially_assignable(DerivedFromRefQualifier&&, const DerivedFromRefQualifier&), "");
static_assert(!__is_trivially_assignable(DerivedFromRefQualifier&, DerivedFromRefQualifier&&), "");
static_assert(!__is_trivially_assignable(DerivedFromRefQualifier&&, DerivedFromRefQualifier&&), "");

struct TemplateAssignNoMove {
  TemplateAssignNoMove &operator=(const TemplateAssignNoMove &) = default;
  template<typename T> TemplateAssignNoMove &operator=(T &&);
};
static_assert(__is_trivially_assignable(TemplateAssignNoMove, const TemplateAssignNoMove &), "");
static_assert(!__is_trivially_assignable(TemplateAssignNoMove, TemplateAssignNoMove &&), "");

struct UseTemplateAssignNoMove {
  TemplateAssignNoMove tanm;
};
static_assert(__is_trivially_assignable(UseTemplateAssignNoMove, const UseTemplateAssignNoMove &), "");
static_assert(!__is_trivially_assignable(UseTemplateAssignNoMove, UseTemplateAssignNoMove &&), "");

struct TemplateAssignNoMoveSFINAE {
  TemplateAssignNoMoveSFINAE &operator=(const TemplateAssignNoMoveSFINAE &) = default;
  template<typename T, typename U = typename T::error> TemplateAssignNoMoveSFINAE &operator=(T &&);
};
static_assert(__is_trivially_assignable(TemplateAssignNoMoveSFINAE, const TemplateAssignNoMoveSFINAE &), "");
static_assert(__is_trivially_assignable(TemplateAssignNoMoveSFINAE, TemplateAssignNoMoveSFINAE &&), "");

struct UseTemplateAssignNoMoveSFINAE {
  TemplateAssignNoMoveSFINAE tanm;
};
static_assert(__is_trivially_assignable(UseTemplateAssignNoMoveSFINAE, const UseTemplateAssignNoMoveSFINAE &), "");
static_assert(__is_trivially_assignable(UseTemplateAssignNoMoveSFINAE, UseTemplateAssignNoMoveSFINAE &&), "");

namespace TrivialityDependsOnImplicitDeletion {
  struct PrivateMove {
    PrivateMove &operator=(const PrivateMove &) = default;
  private:
    PrivateMove &operator=(PrivateMove &&);
    friend class Access;
  };
  static_assert(__is_trivially_assignable(PrivateMove, const PrivateMove &), "");
  static_assert(!__is_trivially_assignable(PrivateMove, PrivateMove &&), "");

  struct NoAccess {
    PrivateMove pm;
    // NoAccess's move would be deleted, so is suppressed,
    // so moves of it use PrivateMove's copy ctor, which is trivial.
  };
  static_assert(__is_trivially_assignable(NoAccess, const NoAccess &), "");
  static_assert(__is_trivially_assignable(NoAccess, NoAccess &&), "");
  struct TopNoAccess : NoAccess {};
  static_assert(__is_trivially_assignable(TopNoAccess, const TopNoAccess &), "");
  static_assert(__is_trivially_assignable(TopNoAccess, TopNoAccess &&), "");

  struct Access {
    PrivateMove pm;
    // NoAccess's move would *not* be deleted, so is *not* suppressed,
    // so moves of it use PrivateMove's move ctor, which is not trivial.
  };
  static_assert(__is_trivially_assignable(Access, const Access &), "");
  static_assert(!__is_trivially_assignable(Access, Access &&), "");
  struct TopAccess : Access {};
  static_assert(__is_trivially_assignable(TopAccess, const TopAccess &), "");
  static_assert(!__is_trivially_assignable(TopAccess, TopAccess &&), "");
}
