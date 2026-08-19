// RUN: %clang_cc1 -std=c++03 -fsyntax-only -verify %s -triple x86_64-windows-msvc
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s -triple x86_64-windows-msvc
// RUN: %clang_cc1 -std=c++03 -fsyntax-only -verify %s -triple x86_64-apple-darwin10
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s -triple x86_64-apple-darwin10

// expected-no-diagnostics

#if __cplusplus < 201103L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__, "")
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

template <class T>
struct Agg {
  T t_;
};

template <class T>
struct Der : T {
};

template <class T>
struct Mut {
  mutable T t_;
};

template <class T>
struct Non {
  Non(); // make it a non-aggregate
  T t_;
};

struct CompletelyTrivial {
};
static_assert(__is_trivially_relocatable(CompletelyTrivial));
static_assert(__is_trivially_relocatable(Agg<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Der<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Mut<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Non<CompletelyTrivial>));

struct NonTrivialDtor {
  ~NonTrivialDtor();
};
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(__is_trivially_relocatable(NonTrivialDtor)); // bug #69394
static_assert(__is_trivially_relocatable(Agg<NonTrivialDtor>));
static_assert(__is_trivially_relocatable(Der<NonTrivialDtor>));
static_assert(__is_trivially_relocatable(Mut<NonTrivialDtor>));
static_assert(__is_trivially_relocatable(Non<NonTrivialDtor>));
#else
static_assert(!__is_trivially_relocatable(NonTrivialDtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialDtor>));
#endif

struct NonTrivialCopyCtor {
  NonTrivialCopyCtor(const NonTrivialCopyCtor&);
};
static_assert(!__is_trivially_relocatable(NonTrivialCopyCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialCopyCtor>));

struct NonTrivialMutableCopyCtor {
  NonTrivialMutableCopyCtor(NonTrivialMutableCopyCtor&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMutableCopyCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMutableCopyCtor>));

#if __cplusplus >= 201103L
struct NonTrivialMoveCtor {
  NonTrivialMoveCtor(NonTrivialMoveCtor&&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMoveCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMoveCtor>));
#endif

struct NonTrivialCopyAssign {
  NonTrivialCopyAssign& operator=(const NonTrivialCopyAssign&);
};
static_assert(__is_trivially_relocatable(NonTrivialCopyAssign));
static_assert(__is_trivially_relocatable(Agg<NonTrivialCopyAssign>));
static_assert(__is_trivially_relocatable(Der<NonTrivialCopyAssign>));
static_assert(__is_trivially_relocatable(Mut<NonTrivialCopyAssign>));
static_assert(__is_trivially_relocatable(Non<NonTrivialCopyAssign>));

struct NonTrivialMutableCopyAssign {
  NonTrivialMutableCopyAssign& operator=(NonTrivialMutableCopyAssign&);
};
static_assert(__is_trivially_relocatable(NonTrivialMutableCopyAssign));
static_assert(__is_trivially_relocatable(Agg<NonTrivialMutableCopyAssign>));
static_assert(__is_trivially_relocatable(Der<NonTrivialMutableCopyAssign>));
static_assert(__is_trivially_relocatable(Mut<NonTrivialMutableCopyAssign>));
static_assert(__is_trivially_relocatable(Non<NonTrivialMutableCopyAssign>));

#if __cplusplus >= 201103L
struct NonTrivialMoveAssign {
  NonTrivialMoveAssign& operator=(NonTrivialMoveAssign&&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMoveAssign));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMoveAssign>));
#endif

struct ImplicitlyDeletedAssign {
  int& r;
};
static_assert(__is_trivially_relocatable(ImplicitlyDeletedAssign));
static_assert(__is_trivially_relocatable(Agg<ImplicitlyDeletedAssign>));
static_assert(__is_trivially_relocatable(Der<ImplicitlyDeletedAssign>));
static_assert(__is_trivially_relocatable(Mut<ImplicitlyDeletedAssign>));
static_assert(__is_trivially_relocatable(Non<ImplicitlyDeletedAssign>));

#if __cplusplus >= 201103L
struct DeletedCopyAssign {
  DeletedCopyAssign(const DeletedCopyAssign&) = default;
  DeletedCopyAssign& operator=(const DeletedCopyAssign&) = delete;
  ~DeletedCopyAssign() = default;
};
static_assert(__is_trivially_relocatable(DeletedCopyAssign));
static_assert(__is_trivially_relocatable(Agg<DeletedCopyAssign>));
static_assert(__is_trivially_relocatable(Der<DeletedCopyAssign>));
static_assert(__is_trivially_relocatable(Mut<DeletedCopyAssign>));
static_assert(__is_trivially_relocatable(Non<DeletedCopyAssign>));

struct DeletedMoveAssign {
  DeletedMoveAssign(DeletedMoveAssign&&) = default;
  DeletedMoveAssign& operator=(DeletedMoveAssign&&) = delete;
  ~DeletedMoveAssign() = default;
};
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(!__is_trivially_relocatable(DeletedMoveAssign)); // bug #69394
static_assert(!__is_trivially_relocatable(Agg<DeletedMoveAssign>));
static_assert(!__is_trivially_relocatable(Der<DeletedMoveAssign>));
static_assert(!__is_trivially_relocatable(Mut<DeletedMoveAssign>));
static_assert(!__is_trivially_relocatable(Non<DeletedMoveAssign>));
#else
static_assert(__is_trivially_relocatable(DeletedMoveAssign));
static_assert(__is_trivially_relocatable(Agg<DeletedMoveAssign>));
static_assert(__is_trivially_relocatable(Der<DeletedMoveAssign>));
static_assert(__is_trivially_relocatable(Mut<DeletedMoveAssign>));
static_assert(__is_trivially_relocatable(Non<DeletedMoveAssign>));
#endif

struct DeletedDestructor {
  DeletedDestructor();
  ~DeletedDestructor() = delete;
};
static_assert(__is_trivially_relocatable(DeletedDestructor)); // bug #38398
static_assert(!__is_trivially_relocatable(Agg<DeletedDestructor>));
static_assert(!__is_trivially_relocatable(Der<DeletedDestructor>));
static_assert(!__is_trivially_relocatable(Mut<DeletedDestructor>));
static_assert(!__is_trivially_relocatable(Non<DeletedDestructor>));
#endif

#if __cplusplus >= 202002L
template<bool B>
struct EligibleNonTrivialDefaultCtor {
    EligibleNonTrivialDefaultCtor() requires B;
    EligibleNonTrivialDefaultCtor() = default;
};
// Only the Rule of 5 members (not default ctor) affect trivial relocatability.
static_assert(__is_trivially_relocatable(EligibleNonTrivialDefaultCtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialDefaultCtor<false>));

template<bool B>
struct IneligibleNonTrivialDefaultCtor {
    IneligibleNonTrivialDefaultCtor();
    IneligibleNonTrivialDefaultCtor() requires B = default;
};
// Only the Rule of 5 members (not default ctor) affect trivial relocatability.
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDefaultCtor<true>));
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDefaultCtor<false>));

template<bool B>
struct EligibleNonTrivialCopyCtor {
    EligibleNonTrivialCopyCtor(const EligibleNonTrivialCopyCtor&) requires B;
    EligibleNonTrivialCopyCtor(const EligibleNonTrivialCopyCtor&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialCopyCtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialCopyCtor<false>));

template<bool B>
struct IneligibleNonTrivialCopyCtor {
    IneligibleNonTrivialCopyCtor(const IneligibleNonTrivialCopyCtor&);
    IneligibleNonTrivialCopyCtor(const IneligibleNonTrivialCopyCtor&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialCopyCtor<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialCopyCtor<false>));

template<bool B>
struct EligibleNonTrivialMoveCtor {
    EligibleNonTrivialMoveCtor(EligibleNonTrivialMoveCtor&&) requires B;
    EligibleNonTrivialMoveCtor(EligibleNonTrivialMoveCtor&&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveCtor<true>));
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveCtor<false>)); // bug #69394
#else
static_assert(__is_trivially_relocatable(EligibleNonTrivialMoveCtor<false>));
#endif

template<bool B>
struct IneligibleNonTrivialMoveCtor {
    IneligibleNonTrivialMoveCtor(IneligibleNonTrivialMoveCtor&&);
    IneligibleNonTrivialMoveCtor(IneligibleNonTrivialMoveCtor&&) requires B = default;
};
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveCtor<true>)); // bug #69394
#else
static_assert(__is_trivially_relocatable(IneligibleNonTrivialMoveCtor<true>));
#endif
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveCtor<false>));

template<bool B>
struct EligibleNonTrivialCopyAssign {
    EligibleNonTrivialCopyAssign& operator=(const EligibleNonTrivialCopyAssign&) requires B;
    EligibleNonTrivialCopyAssign& operator=(const EligibleNonTrivialCopyAssign&) = default;
};
static_assert(__is_trivially_relocatable(EligibleNonTrivialCopyAssign<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialCopyAssign<false>));

template<bool B>
struct IneligibleNonTrivialCopyAssign {
    IneligibleNonTrivialCopyAssign& operator=(const IneligibleNonTrivialCopyAssign&);
    IneligibleNonTrivialCopyAssign& operator=(const IneligibleNonTrivialCopyAssign&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialCopyAssign<true>));
static_assert(__is_trivially_relocatable(IneligibleNonTrivialCopyAssign<false>));

template<bool B>
struct EligibleNonTrivialMoveAssign {
    EligibleNonTrivialMoveAssign& operator=(EligibleNonTrivialMoveAssign&&) requires B;
    EligibleNonTrivialMoveAssign& operator=(EligibleNonTrivialMoveAssign&&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveAssign<true>));
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveAssign<false>));

template<bool B>
struct IneligibleNonTrivialMoveAssign {
    IneligibleNonTrivialMoveAssign& operator=(IneligibleNonTrivialMoveAssign&&);
    IneligibleNonTrivialMoveAssign& operator=(IneligibleNonTrivialMoveAssign&&) requires B = default;
};
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveAssign<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveAssign<false>));

template<bool B>
struct EligibleNonTrivialDtor {
    ~EligibleNonTrivialDtor() requires B;
    ~EligibleNonTrivialDtor() = default;
};
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(__is_trivially_relocatable(EligibleNonTrivialDtor<true>)); // bug #69394
#else
static_assert(!__is_trivially_relocatable(EligibleNonTrivialDtor<true>));
#endif
static_assert(__is_trivially_relocatable(EligibleNonTrivialDtor<false>));

template<bool B>
struct IneligibleNonTrivialDtor {
    ~IneligibleNonTrivialDtor();
    ~IneligibleNonTrivialDtor() requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDtor<true>));
#if defined(_WIN64) && !defined(__MINGW32__)
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDtor<false>)); // bug #69394
#else
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialDtor<false>));
#endif
#endif

#if __cplusplus >= 201103L
namespace MutableMembers {
  // Make sure Clang properly handles these two tricky cases.
  // The copy constructor of MutableFieldUsesNonTrivialCopyCtor
  // uses a non-trivial constructor of TMovableButNonTCopyable,
  // so MutableFieldUsesNonTrivialCopyCtor's copy constructor is
  // not trivial.

  struct TMovableButNonTCopyable {
    TMovableButNonTCopyable(TMovableButNonTCopyable&); // user-provided
    TMovableButNonTCopyable(TMovableButNonTCopyable&&) = default;
    TMovableButNonTCopyable& operator=(TMovableButNonTCopyable&&) = default;
    ~TMovableButNonTCopyable() = default;
  };
  static_assert(!__is_trivially_copyable(TMovableButNonTCopyable), "");
  static_assert(!__is_trivially_relocatable(TMovableButNonTCopyable), "");

  struct MutableFieldUsesNonTrivialCopyCtor {
    mutable TMovableButNonTCopyable m;
  };
  static_assert(!__is_trivially_copyable(MutableFieldUsesNonTrivialCopyCtor), "");
  static_assert(!__is_trivially_relocatable(MutableFieldUsesNonTrivialCopyCtor), "");

  // The copy constructor of MutableFieldIsMoveOnly is implicitly
  // deleted, which makes MutableFieldIsMoveOnly trivially copyable
  // and therefore also trivially relocatable.

  struct TMovableButNotCopyable {
    TMovableButNotCopyable(TMovableButNotCopyable&) = delete;
    TMovableButNotCopyable(TMovableButNotCopyable&&) = default;
    TMovableButNotCopyable& operator=(TMovableButNotCopyable&&) = default;
    ~TMovableButNotCopyable() = default;
  };
  static_assert(__is_trivially_copyable(TMovableButNotCopyable));
#if defined(_WIN64) && !defined(__MINGW32__)
  static_assert(!__is_trivially_relocatable(TMovableButNotCopyable)); // bug #69394
#else
  static_assert(__is_trivially_relocatable(TMovableButNotCopyable));
#endif
  struct MutableFieldIsMoveOnly {
    mutable TMovableButNotCopyable m;
  };
  static_assert(__is_trivially_copyable(MutableFieldIsMoveOnly));
#if defined(_WIN64) && !defined(__MINGW32__)
  static_assert(!__is_trivially_relocatable(MutableFieldIsMoveOnly)); // bug #69394
#else
  static_assert(__is_trivially_relocatable(MutableFieldIsMoveOnly));
#endif
}
#endif
