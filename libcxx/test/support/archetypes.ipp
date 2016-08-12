
#ifndef DEFINE_EXPLICIT
#define DEFINE_EXPLICIT
#endif
#ifndef DEFINE_CTOR
#define DEFINE_CTOR = default
#endif
#ifndef DEFINE_ASSIGN
#define DEFINE_ASSIGN = default
#endif
#ifndef DEFINE_DTOR
#define DEFINE_DTOR(Name)
#endif

struct NoDefault {
  DEFINE_EXPLICIT NoDefault() = delete;
  DEFINE_DTOR(NoDefault)
};

struct AllCtors {
  DEFINE_EXPLICIT AllCtors() DEFINE_CTOR;
  DEFINE_EXPLICIT AllCtors(AllCtors const&) DEFINE_CTOR;
  DEFINE_EXPLICIT AllCtors(AllCtors &&) DEFINE_CTOR;
  AllCtors& operator=(AllCtors const&) DEFINE_ASSIGN;
  AllCtors& operator=(AllCtors &&) DEFINE_ASSIGN;
  DEFINE_DTOR(AllCtors)
};

struct Copyable {
  DEFINE_EXPLICIT Copyable() DEFINE_CTOR;
  DEFINE_EXPLICIT Copyable(Copyable const &) DEFINE_CTOR;
  Copyable &operator=(Copyable const &) DEFINE_ASSIGN;
  DEFINE_DTOR(Copyable)
};

struct CopyOnly {
  DEFINE_EXPLICIT CopyOnly() DEFINE_CTOR;
  DEFINE_EXPLICIT CopyOnly(CopyOnly const &) DEFINE_CTOR;
  DEFINE_EXPLICIT CopyOnly(CopyOnly &&) = delete;
  CopyOnly &operator=(CopyOnly const &) DEFINE_ASSIGN;
  CopyOnly &operator=(CopyOnly &&) = delete;
  DEFINE_DTOR(CopyOnly)
};

struct NonCopyable {
  DEFINE_EXPLICIT NonCopyable() DEFINE_CTOR;
  DEFINE_EXPLICIT NonCopyable(NonCopyable const &) = delete;
  NonCopyable &operator=(NonCopyable const &) = delete;
  DEFINE_DTOR(NonCopyable)
};

struct MoveOnly {
  DEFINE_EXPLICIT MoveOnly() DEFINE_CTOR;
  DEFINE_EXPLICIT MoveOnly(MoveOnly &&) DEFINE_CTOR;
  MoveOnly &operator=(MoveOnly &&) DEFINE_ASSIGN;
  DEFINE_DTOR(MoveOnly)
};

struct CopyAssignable {
    DEFINE_EXPLICIT CopyAssignable() = delete;
    CopyAssignable& operator=(CopyAssignable const&) DEFINE_ASSIGN;
    DEFINE_DTOR(CopyAssignable)
};

struct CopyAssignOnly {
    DEFINE_EXPLICIT CopyAssignOnly() = delete;
    CopyAssignOnly& operator=(CopyAssignOnly const&) DEFINE_ASSIGN;
    CopyAssignOnly& operator=(CopyAssignOnly &&) = delete;
    DEFINE_DTOR(CopyAssignOnly)
};

struct MoveAssignOnly {
    DEFINE_EXPLICIT MoveAssignOnly() = delete;
    MoveAssignOnly& operator=(MoveAssignOnly const&) = delete;
    MoveAssignOnly& operator=(MoveAssignOnly &&) DEFINE_ASSIGN;
    DEFINE_DTOR(MoveAssignOnly)
};

struct ConvertingType {
  DEFINE_EXPLICIT ConvertingType() DEFINE_CTOR;
  DEFINE_EXPLICIT ConvertingType(ConvertingType const&) DEFINE_CTOR;
  DEFINE_EXPLICIT ConvertingType(ConvertingType &&) DEFINE_CTOR;
  ConvertingType& operator=(ConvertingType const&) DEFINE_ASSIGN;
  ConvertingType& operator=(ConvertingType &&) DEFINE_ASSIGN;
  template <class ...Args>
  DEFINE_EXPLICIT ConvertingType(Args&&...) {}
  template <class Arg>
  ConvertingType& operator=(Arg&&) { return *this; }
  DEFINE_DTOR(ConvertingType)
};

template <template <class...> class List>
using ApplyTypes = List<
    NoDefault,
    AllCtors,
    Copyable,
    CopyOnly,
    NonCopyable,
    MoveOnly,
    CopyAssignable,
    CopyAssignOnly,
    MoveAssignOnly,
    ConvertingType
  >;

#undef DEFINE_EXPLICIT
#undef DEFINE_CTOR
#undef DEFINE_ASSIGN
#undef DEFINE_DTOR
