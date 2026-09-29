// RUN: %check_clang_tidy %s modernize-use-equals-default %t -- -- -fno-delayed-template-parsing -fexceptions

// Out of line definition.
class OL {
public:
  OL();
  ~OL();
};

OL::OL() {};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default' to define a trivial default constructor [modernize-use-equals-default]
// CHECK-FIXES: OL::OL() = default;
OL::~OL() {}
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default' to define a trivial destructor [modernize-use-equals-default]
// CHECK-FIXES: OL::~OL() = default;

// Inline definitions.
class IL {
public:
  IL() {} 	 ; // Note embedded tab on this line
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: IL() = default 	 ; // Note embedded tab on this line
  ~IL() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~IL() = default;
};

// Non-empty body.
void f();
class NE {
public:
  NE() { f(); }
  ~NE() { f(); }
};

// Skip unions.
union NU {
  NU() {}
  // CHECK-FIXES: NU() {}
  ~NU() {}
  // CHECK-FIXES: ~NU() {}
  NE Field;
};

// Skip unions with out-of-line constructor/destructor.
union NUO {
  NUO();
  ~NUO();
  NE Field;
};

NUO::NUO() {}
// CHECK-FIXES: NUO::NUO() {}
NUO::~NUO() {}
// CHECK-FIXES: NUO::~NUO() {}

// Skip structs/classes containing anonymous unions.
struct SU {
  SU() {}
  // CHECK-FIXES: SU() {}
  ~SU() {}
  // CHECK-FIXES: ~SU() {}
  union {
    NE Field;
  };
};

// Skip variadic constructors.
struct VA {
  VA(...) {}
};

// Skip template constructors.
struct TC {
  template <unsigned U>
  TC() {}

  template <unsigned U>
  TC(const TC &) {}

  template <unsigned U>
  TC& operator = (const TC &) { return *this; }
};

// Initializer or arguments.
class IA {
public:
  // Constructor with initializer.
  IA() : Field(5) {}
  // Constructor with arguments.
  IA(int Arg1, int Arg2) {}
  int Field;
};

// Default member initializer
class DMI {
public:
  DMI() {} // Comment before semi-colon on next line
  ;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use '= default'
  // CHECK-FIXES: DMI() = default // Comment before semi-colon on next line
  // CHECK-FIXES-NEXT:   ;
  int Field = 5;
};

// Class member
class CM {
public:
  CM() {} /* Comments */ /* before */ /* semicolon */;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: CM() = default /* Comments */ /* before */ /* semicolon */;
  OL o;
};

// Private constructor/destructor.
class Priv {
  Priv();
  ~Priv() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~Priv() = default;
};

Priv::Priv() {}
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use '= default'
// CHECK-FIXES: Priv::Priv() = default;

struct SemiColon {
  SemiColon() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: SemiColon() = default;
};

struct SemiColonOutOfLine {
  SemiColonOutOfLine();
};

SemiColonOutOfLine::SemiColonOutOfLine() {};
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use '= default'
// CHECK-FIXES: SemiColonOutOfLine::SemiColonOutOfLine() = default;

// struct.
struct ST {
  ST() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ST() = default;
  ~ST() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~ST() = default;
};

// Deleted constructor/destructor.
class Del {
public:
  Del() = delete;
  ~Del() = delete;
};

// Do not remove other keywords.
class KW {
public:
  explicit KW() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use '= default'
  // CHECK-FIXES: explicit KW() = default;
  virtual ~KW() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use '= default'
  // CHECK-FIXES: virtual ~KW() = default;
};

// Nested class.
struct N {
  struct NN {
    NN() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default'
    // CHECK-FIXES: NN() = default;
    ~NN() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default'
    // CHECK-FIXES: ~NN() = default;
  };
  int Int;
};

// Class template.
template <class T>
class Temp {
public:
  Temp() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: Temp() = default;
  ~Temp() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~Temp() = default;
};

// Class template out of line with explicit instantiation.
template <class T>
class TempODef {
public:
  TempODef();
  ~TempODef();
};

template <class T>
TempODef<T>::TempODef() {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use '= default'
// CHECK-FIXES: TempODef<T>::TempODef() = default;
template <class T>
TempODef<T>::~TempODef() {}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use '= default'
// CHECK-FIXES: TempODef<T>::~TempODef() = default;

template class TempODef<int>;
template class TempODef<double>;

// Non user-provided constructor/destructor.
struct Imp {
  int Int;
};
void g() {
  Imp *PtrImp = new Imp();
  PtrImp->~Imp();
  delete PtrImp;
}

// Already using default.
struct IDef {
  IDef() = default;
  ~IDef() = default;
};
struct ODef {
  ODef();
  ~ODef();
};
ODef::ODef() = default;
ODef::~ODef() = default;

// Delegating constructor and overriden destructor.
struct DC : KW {
  DC() : KW() {}
  ~DC() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~DC() override = default;
};

struct OverrideWithSemiColon : KW {
  ~OverrideWithSemiColon() override {};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~OverrideWithSemiColon() override = default;
};

struct Comments {
  Comments() {
    // Don't erase comments inside the body.
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use '= default'
  ~Comments() {
    // Don't erase comments inside the body.
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use '= default'
};

// Try-catch.
struct ITC {
  ITC() try {} catch(...) {}
  ~ITC() try {} catch(...) {}
};

struct OTC {
  OTC();
  ~OTC();
};
OTC::OTC() try {} catch(...) {}
OTC::~OTC() try {} catch(...) {}

#define STRUCT_WITH_DEFAULT(_base, _type) \
  struct _type {                          \
    _type() {}                            \
    _base value;                          \
  };

STRUCT_WITH_DEFAULT(unsigned char, InMacro)

#define ZERO_VALUE 0
struct PreprocesorDependentTest
{
  void something();

  PreprocesorDependentTest() {
#if ZERO_VALUE
    something();
#endif
  }

  ~PreprocesorDependentTest() {
#if ZERO_VALUE
    something();
#endif
  }
};
