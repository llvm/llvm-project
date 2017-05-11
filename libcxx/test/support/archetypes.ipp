
#ifndef DEFINE_BASE
#define DEFINE_BASE(Name) ::ArchetypeBases::NullBase
#endif
#ifndef DEFINE_EXPLICIT
#define DEFINE_EXPLICIT
#endif
#ifndef DEFINE_CONSTEXPR
#ifdef TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#define DEFINE_CONSTEXPR
#else // TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#define DEFINE_CONSTEXPR constexpr
#endif // TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#endif
#ifndef DEFINE_ASSIGN_CONSTEXPR
#if TEST_STD_VER >= 14
#define DEFINE_ASSIGN_CONSTEXPR DEFINE_CONSTEXPR
#else
#define DEFINE_ASSIGN_CONSTEXPR
#endif
#endif
#ifndef DEFINE_CTOR
#define DEFINE_CTOR = default
#endif
#ifndef DEFINE_DEFAULT_CTOR
#define DEFINE_DEFAULT_CTOR DEFINE_CTOR
#endif
#ifndef DEFINE_ASSIGN
#define DEFINE_ASSIGN = default
#endif
#ifndef DEFINE_DTOR
#define DEFINE_DTOR(Name)
#endif

struct AllCtors : DEFINE_BASE(AllCtors) {
  using Base = DEFINE_BASE(AllCtors);
  using Base::Base;
  using Base::operator=;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR AllCtors() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR AllCtors(AllCtors const&) DEFINE_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR AllCtors(AllCtors &&) DEFINE_CTOR;
  DEFINE_ASSIGN_CONSTEXPR AllCtors& operator=(AllCtors const&) DEFINE_ASSIGN;
  DEFINE_ASSIGN_CONSTEXPR AllCtors& operator=(AllCtors &&) DEFINE_ASSIGN;
  DEFINE_DTOR(AllCtors)
};

struct NoCtors : DEFINE_BASE(NoCtors) {
  using Base = DEFINE_BASE(NoCtors);
  DEFINE_EXPLICIT NoCtors() = delete;
  DEFINE_EXPLICIT NoCtors(NoCtors const&) = delete;
  NoCtors& operator=(NoCtors const&) = delete;
  DEFINE_DTOR(NoCtors)
};

struct NoDefault : DEFINE_BASE(NoDefault) {
  using Base = DEFINE_BASE(NoDefault);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR NoDefault() = delete;
  DEFINE_DTOR(NoDefault)
};

struct DefaultOnly : DEFINE_BASE(DefaultOnly) {
  using Base = DEFINE_BASE(DefaultOnly);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR DefaultOnly() DEFINE_DEFAULT_CTOR;
  DefaultOnly(DefaultOnly const&) = delete;
  DefaultOnly& operator=(DefaultOnly const&) = delete;
  DEFINE_DTOR(DefaultOnly)
};

struct Copyable : DEFINE_BASE(Copyable) {
  using Base = DEFINE_BASE(Copyable);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR Copyable() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR Copyable(Copyable const &) DEFINE_CTOR;
  Copyable &operator=(Copyable const &) DEFINE_ASSIGN;
  DEFINE_DTOR(Copyable)
};

struct CopyOnly : DEFINE_BASE(CopyOnly) {
  using Base = DEFINE_BASE(CopyOnly);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyOnly() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyOnly(CopyOnly const &) DEFINE_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyOnly(CopyOnly &&) = delete;
  CopyOnly &operator=(CopyOnly const &) DEFINE_ASSIGN;
  CopyOnly &operator=(CopyOnly &&) = delete;
  DEFINE_DTOR(CopyOnly)
};

struct NonCopyable : DEFINE_BASE(NonCopyable) {
  using Base = DEFINE_BASE(NonCopyable);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR NonCopyable() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR NonCopyable(NonCopyable const &) = delete;
  NonCopyable &operator=(NonCopyable const &) = delete;
  DEFINE_DTOR(NonCopyable)
};

struct MoveOnly : DEFINE_BASE(MoveOnly) {
  using Base = DEFINE_BASE(MoveOnly);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR MoveOnly() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR MoveOnly(MoveOnly &&) DEFINE_CTOR;
  MoveOnly &operator=(MoveOnly &&) DEFINE_ASSIGN;
  DEFINE_DTOR(MoveOnly)
};

struct CopyAssignable : DEFINE_BASE(CopyAssignable) {
  using Base = DEFINE_BASE(CopyAssignable);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyAssignable() = delete;
  CopyAssignable& operator=(CopyAssignable const&) DEFINE_ASSIGN;
  DEFINE_DTOR(CopyAssignable)
};

struct CopyAssignOnly : DEFINE_BASE(CopyAssignOnly) {
  using Base = DEFINE_BASE(CopyAssignOnly);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyAssignOnly() = delete;
  CopyAssignOnly& operator=(CopyAssignOnly const&) DEFINE_ASSIGN;
  CopyAssignOnly& operator=(CopyAssignOnly &&) = delete;
  DEFINE_DTOR(CopyAssignOnly)
};

struct MoveAssignOnly : DEFINE_BASE(MoveAssignOnly) {
  using Base = DEFINE_BASE(MoveAssignOnly);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR MoveAssignOnly() = delete;
  MoveAssignOnly& operator=(MoveAssignOnly const&) = delete;
  MoveAssignOnly& operator=(MoveAssignOnly &&) DEFINE_ASSIGN;
  DEFINE_DTOR(MoveAssignOnly)
};

struct ConvertingType : DEFINE_BASE(ConvertingType) {
  using Base = DEFINE_BASE(ConvertingType);
  using Base::Base;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType() DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(ConvertingType const&) DEFINE_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(ConvertingType &&) DEFINE_CTOR;
  ConvertingType& operator=(ConvertingType const&) DEFINE_ASSIGN;
  ConvertingType& operator=(ConvertingType &&) DEFINE_ASSIGN;
  template <class ...Args>
  DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(Args&&...) {}
  template <class Arg>
  ConvertingType& operator=(Arg&&) { return *this; }
  DEFINE_DTOR(ConvertingType)
};

template <template <class...> class List>
using ApplyTypes = List<
    AllCtors,
    NoCtors,
    NoDefault,
    DefaultOnly,
    Copyable,
    CopyOnly,
    NonCopyable,
    MoveOnly,
    CopyAssignable,
    CopyAssignOnly,
    MoveAssignOnly,
    ConvertingType
  >;

#undef DEFINE_BASE
#undef DEFINE_EXPLICIT
#undef DEFINE_CONSTEXPR
#undef DEFINE_ASSIGN_CONSTEXPR
#undef DEFINE_CTOR
#undef DEFINE_DEFAULT_CTOR
#undef DEFINE_ASSIGN
#undef DEFINE_DTOR
