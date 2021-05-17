// RUN: %check_clang_tidy %s modernize-use-equals-default %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-equals-default.IgnoreMacros, value: false}]}" \
// RUN:   -- -fno-delayed-template-parsing -fexceptions

// Out of line definition.
struct OL {
  OL(const OL &);
  OL &operator=(const OL &);
  int Field;
};
OL::OL(const OL &Other) : Field(Other.Field) {}
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default' to define a trivial copy constructor [modernize-use-equals-default]
// CHECK-FIXES: OL::OL(const OL &Other)  = default;
OL &OL::operator=(const OL &Other) {
  Field = Other.Field;
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-4]]:9: warning: use '= default' to define a trivial copy-assignment operator [modernize-use-equals-default]
// CHECK-FIXES: OL &OL::operator=(const OL &Other) = default;

// Inline.
struct IL {
  IL(const IL &Other) : Field(Other.Field) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: IL(const IL &Other)  = default;
  IL &operator=(const IL &Other) {
    Field = Other.Field;
    return *this;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: use '= default'
  // CHECK-FIXES: IL &operator=(const IL &Other) = default;
  int Field;
};

// Wrong type.
struct WT {
  WT(const IL &Other) {}
  WT &operator=(const IL &);
};
WT &WT::operator=(const IL &Other) { return *this; }

// Qualifiers.
struct Qual {
  Qual(const Qual &Other) : Field(Other.Field), Volatile(Other.Volatile),
                            Mutable(Other.Mutable), Reference(Other.Reference),
                            Const(Other.Const) {}
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use '= default'
  // CHECK-FIXES: Qual(const Qual &Other)
  // CHECK-FIXES:                          = default;

  int Field;
  volatile char Volatile;
  mutable bool Mutable;
  const OL &Reference; // This makes this class non-assignable.
  const IL Const;      // This also makes this class non-assignable.
  static int Static;
};

// Wrong init arguments.
struct WI {
  WI(const WI &Other) : Field1(Other.Field1), Field2(Other.Field1) {}
  WI &operator=(const WI &);
  int Field1, Field2;
};
WI &WI::operator=(const WI &Other) {
  Field1 = Other.Field1;
  Field2 = Other.Field1;
  return *this;
}

// Missing field.
struct MF {
  MF(const MF &Other) : Field1(Other.Field1), Field2(Other.Field2) {}
  MF &operator=(const MF &);
  int Field1, Field2, Field3;
};
MF &MF::operator=(const MF &Other) {
  Field1 = Other.Field1;
  Field2 = Other.Field2;
  return *this;
}

struct Comments {
  Comments(const Comments &Other)
      /* don't delete */ : /* this comment */ Field(Other.Field) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use '= default'
  // CHECK-FIXES: /* don't delete */  = default;
  int Field;
};

struct MoreComments {
  MoreComments(const MoreComments &Other) /* this comment is OK */
      : Field(Other.Field) {}
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use '= default'
  // CHECK-FIXES: MoreComments(const MoreComments &Other) /* this comment is OK */
  // CHECK-FIXES-NEXT: = default;
  int Field;
};

struct ColonInComment {
  ColonInComment(const ColonInComment &Other) /* : */ : Field(Other.Field) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ColonInComment(const ColonInComment &Other) /* : */  = default;
  int Field;
};

// No members or bases (in particular, no colon).
struct Empty {
  Empty(const Empty &Other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: Empty(const Empty &Other) = default;
  Empty &operator=(const Empty &);
};
Empty &Empty::operator=(const Empty &Other) { return *this; }
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use '= default'
// CHECK-FIXES: Empty &Empty::operator=(const Empty &Other) = default;

// Bit fields.
struct BF {
  BF() = default;
  BF(const BF &Other) : Field1(Other.Field1), Field2(Other.Field2), Field3(Other.Field3),
                        Field4(Other.Field4) {};
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use '= default'
  // CHECK-FIXES: BF(const BF &Other) {{$}}
  // CHECK-FIXES:                     = default;
  BF &operator=(const BF &);

  unsigned Field1 : 3;
  int : 7;
  char Field2 : 6;
  int : 0;
  int Field3 : 24;
  unsigned char Field4;
};
BF &BF::operator=(const BF &Other) {
  Field1 = Other.Field1;
  Field2 = Other.Field2;
  Field3 = Other.Field3;
  Field4 = Other.Field4;
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-7]]:9: warning: use '= default'
// CHECK-FIXES: BF &BF::operator=(const BF &Other) = default;

// Base classes.
struct BC : IL, OL, BF {
  BC(const BC &Other) : IL(Other), OL(Other), BF(Other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: BC(const BC &Other)  = default;
  BC &operator=(const BC &Other);
};
BC &BC::operator=(const BC &Other) {
  IL::operator=(Other);
  OL::operator=(Other);
  BF::operator=(Other);
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-6]]:9: warning: use '= default'
// CHECK-FIXES: BC &BC::operator=(const BC &Other) = default;

// Base classes with member.
struct BCWM : IL, OL {
  BCWM(const BCWM &Other) : IL(Other), OL(Other), Bf(Other.Bf) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: BCWM(const BCWM &Other)  = default;
  BCWM &operator=(const BCWM &);
  BF Bf;
};
BCWM &BCWM::operator=(const BCWM &Other) {
  IL::operator=(Other);
  OL::operator=(Other);
  Bf = Other.Bf;
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-6]]:13: warning: use '= default'
// CHECK-FIXES: BCWM &BCWM::operator=(const BCWM &Other) = default;

// Missing base class.
struct MBC : IL, OL, BF {
  MBC(const MBC &Other) : IL(Other), OL(Other) {}
  MBC &operator=(const MBC &);
};
MBC &MBC::operator=(const MBC &Other) {
  IL::operator=(Other);
  OL::operator=(Other);
  return *this;
}

// Base classes, incorrect parameter.
struct BCIP : BCWM, BF {
  BCIP(const BCIP &Other) : BCWM(Other), BF(Other.Bf) {}
  BCIP &operator=(const BCIP &);
};
BCIP &BCIP::operator=(const BCIP &Other) {
  BCWM::operator=(Other);
  BF::operator=(Other.Bf);
  return *this;
}

// Virtual base classes.
struct VA : virtual OL {};
struct VB : virtual OL {};
struct VBC : VA, VB, virtual OL {
  // OL is the first thing that is going to be initialized, despite the fact
  // that it is the last in the list of bases, because it is virtual and there
  // is a virtual OL at the beginning of VA (which is the same).
  VBC(const VBC &Other) : OL(Other), VA(Other), VB(Other) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: VBC(const VBC &Other)  = default;
  VBC &operator=(const VBC &Other);
};
VBC &VBC::operator=(const VBC &Other) {
  OL::operator=(Other);
  VA::operator=(Other);
  VB::operator=(Other);
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-6]]:11: warning: use '= default'
// CHECK-FIXES: VBC &VBC::operator=(const VBC &Other) = default;

// Indirect base.
struct IB : VBC {
  IB(const IB &Other) : OL(Other), VBC(Other) {}
  IB &operator=(const IB &);
};
IB &IB::operator=(const IB &Other) {
  OL::operator=(Other);
  VBC::operator=(Other);
  return *this;
}

// Class template.
template <class T>
struct Template {
  Template() = default;
  Template(const Template &Other) : Field(Other.Field) {}
  Template &operator=(const Template &Other);
  void foo(const T &t);
  int Field;
};
template <class T>
Template<T> &Template<T>::operator=(const Template<T> &Other) {
  Field = Other.Field;
  return *this;
}
Template<int> T1;

// Dependent types.
template <class T>
struct DT1 {
  DT1() = default;
  DT1(const DT1 &Other) : Field(Other.Field) {}
  DT1 &operator=(const DT1 &);
  T Field;
};
template <class T>
DT1<T> &DT1<T>::operator=(const DT1<T> &Other) {
  Field = Other.Field;
  return *this;
}
DT1<int> Dt1;

template <class T>
struct DT2 {
  DT2() = default;
  DT2(const DT2 &Other) : Field(Other.Field), Dependent(Other.Dependent) {}
  DT2 &operator=(const DT2 &);
  T Field;
  typename T::TT Dependent;
};
template <class T>
DT2<T> &DT2<T>::operator=(const DT2<T> &Other) {
  Field = Other.Field;
  Dependent = Other.Dependent;
  return *this;
}
struct T {
  typedef int TT;
};
DT2<T> Dt2;

// Default arguments.
struct DA {
  DA(int Int);
  DA(const DA &Other = DA(0)) : Field1(Other.Field1), Field2(Other.Field2) {}
  DA &operator=(const DA &);
  int Field1;
  char Field2;
};
// Overloaded operator= cannot have a default argument.
DA &DA::operator=(const DA &Other) {
  Field1 = Other.Field1;
  Field2 = Other.Field2;
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: use '= default'
// CHECK-FIXES: DA &DA::operator=(const DA &Other) = default;

struct DA2 {
  // Can be used as copy-constructor but cannot be explicitly defaulted.
  DA2(const DA &Other, int Def = 0) {}
};

// Default initialization.
struct DI {
  DI(const DI &Other) : Field1(Other.Field1), Field2(Other.Field2) {}
  int Field1;
  int Field2 = 0;
  int Fiedl3;
};

// Statement inside body.
void foo();
struct SIB {
  SIB(const SIB &Other) : Field(Other.Field) { foo(); }
  SIB &operator=(const SIB &);
  int Field;
};
SIB &SIB::operator=(const SIB &Other) {
  Field = Other.Field;
  foo();
  return *this;
}

// Comment inside body.
struct CIB {
  CIB(const CIB &Other) : Field(Other.Field) { /* Don't erase this */
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use '= default'
  CIB &operator=(const CIB &);
  int Field;
};
CIB &CIB::operator=(const CIB &Other) {
  Field = Other.Field;
  // FIXME: don't erase this comment.
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-5]]:11: warning: use '= default'
// CHECK-FIXES: CIB &CIB::operator=(const CIB &Other) = default;

// Take non-const reference as argument.
struct NCRef {
  NCRef(NCRef &Other) : Field1(Other.Field1), Field2(Other.Field2) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: NCRef(NCRef &Other)  = default;
  NCRef &operator=(NCRef &);
  int Field1, Field2;
};
NCRef &NCRef::operator=(NCRef &Other) {
  Field1 = Other.Field1;
  Field2 = Other.Field2;
  return *this;
}
// CHECK-MESSAGES: :[[@LINE-5]]:15: warning: use '= default'
// CHECK-FIXES: NCRef &NCRef::operator=(NCRef &Other) = default;

// Already defaulted.
struct IAD {
  IAD(const IAD &Other) = default;
  IAD &operator=(const IAD &Other) = default;
};

struct OAD {
  OAD(const OAD &Other);
  OAD &operator=(const OAD &);
};
OAD::OAD(const OAD &Other) = default;
OAD &OAD::operator=(const OAD &Other) = default;

// Deleted.
struct ID {
  ID(const ID &Other) = delete;
  ID &operator=(const ID &Other) = delete;
};

// Non-reference parameter.
struct NRef {
  NRef &operator=(NRef Other);
  int Field1;
};
NRef &NRef::operator=(NRef Other) {
  Field1 = Other.Field1;
  return *this;
}

// RValue reference parameter.
struct RVR {
  RVR(RVR &&Other) {}
  RVR &operator=(RVR &&);
};
RVR &RVR::operator=(RVR &&Other) { return *this; }

// Similar function.
struct SF {
  SF &foo(const SF &);
  int Field1;
};
SF &SF::foo(const SF &Other) {
  Field1 = Other.Field1;
  return *this;
}

// No return.
struct NR {
  NR &operator=(const NR &);
};
NR &NR::operator=(const NR &Other) {}

// Return misplaced.
struct RM {
  RM &operator=(const RM &);
  int Field;
};
RM &RM::operator=(const RM &Other) {
  return *this;
  Field = Other.Field;
}

// Wrong return value.
struct WRV {
  WRV &operator=(WRV &);
};
WRV &WRV::operator=(WRV &Other) {
  return Other;
}

// Wrong return type.
struct WRT : IL {
  IL &operator=(const WRT &);
};
IL &WRT::operator=(const WRT &Other) {
  return *this;
}

// Try-catch.
struct ITC {
  ITC(const ITC &Other)
  try : Field(Other.Field) {
  } catch (...) {
  }
  ITC &operator=(const ITC &Other) try {
    Field = Other.Field;
  } catch (...) {
  }
  int Field;
};

struct OTC {
  OTC(const OTC &);
  OTC &operator=(const OTC &);
  int Field;
};
OTC::OTC(const OTC &Other) try : Field(Other.Field) {
} catch (...) {
}
OTC &OTC::operator=(const OTC &Other) try {
  Field = Other.Field;
} catch (...) {
}

// FIXME: the check is not able to detect exception specification.
// noexcept(true).
struct NET {
  // This is the default.
  //NET(const NET &Other) noexcept {}
  NET &operator=(const NET &Other) noexcept;
};
//NET &NET::operator=(const NET &Other) noexcept { return *this; }

// noexcept(false).
struct NEF {
  // This is the default.
  //NEF(const NEF &Other) noexcept(false) {}
  NEF &operator=(const NEF &Other) noexcept(false);
};
//NEF &NEF::operator=(const NEF &Other) noexcept(false) { return *this; }

#define STRUCT_WITH_COPY_CONSTRUCT(_base, _type) \
  struct _type {                                 \
    _type(const _type &v) : value(v.value) {}    \
    _base value;                                 \
  };

STRUCT_WITH_COPY_CONSTRUCT(unsigned char, Hex8CopyConstruct)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use '= default' to define a trivial copy constructor
// CHECK-MESSAGES: :[[@LINE-6]]:44: note:

#define STRUCT_WITH_COPY_ASSIGN(_base, _type) \
  struct _type {                              \
    _type &operator=(const _type &rhs) {      \
      value = rhs.value;                      \
      return *this;                           \
    }                                         \
    _base value;                              \
  };

STRUCT_WITH_COPY_ASSIGN(unsigned char, Hex8CopyAssign)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use '= default' to define a trivial copy-assignment operator
// CHECK-MESSAGES: :[[@LINE-9]]:40: note:

// Use of braces
struct UOB{
  UOB(const UOB &Other):j{Other.j}{}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default' to define a trivial copy constructor [modernize-use-equals-default]
  // CHECK-FIXES: UOB(const UOB &Other)= default;
  int j;
};
