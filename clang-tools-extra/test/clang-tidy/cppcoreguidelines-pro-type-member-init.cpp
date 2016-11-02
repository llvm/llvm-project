// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -- -std=c++11 -fno-delayed-template-parsing

struct PositiveFieldBeforeConstructor {
  int F;
  // CHECK-FIXES: int F{};
  PositiveFieldBeforeConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
  // CHECK-FIXES: PositiveFieldBeforeConstructor() {}
};

struct PositiveFieldAfterConstructor {
  PositiveFieldAfterConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F, G
  // CHECK-FIXES: PositiveFieldAfterConstructor() {}
  int F;
  // CHECK-FIXES: int F{};
  bool G /* with comment */;
  // CHECK-FIXES: bool G{} /* with comment */;
  PositiveFieldBeforeConstructor IgnoredField;
};

struct PositiveSeparateDefinition {
  PositiveSeparateDefinition();
  int F;
  // CHECK-FIXES: int F{};
};

PositiveSeparateDefinition::PositiveSeparateDefinition() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: F
// CHECK-FIXES: PositiveSeparateDefinition::PositiveSeparateDefinition() {}

struct PositiveMixedFieldOrder {
  PositiveMixedFieldOrder() : J(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: I, K
  // CHECK-FIXES: PositiveMixedFieldOrder() : J(0) {}
  int I;
  // CHECK-FIXES: int I{};
  int J;
  int K;
  // CHECK-FIXES: int K{};
};

template <typename T>
struct Template {
  Template() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
  int F;
  // CHECK-FIXES: int F{};
  T T1;
  // CHECK-FIXES: T T1;
};

void instantiate() {
  Template<int> TInt;
}

struct NegativeFieldInitialized {
  int F;

  NegativeFieldInitialized() : F() {}
};

struct NegativeFieldInitializedInDefinition {
  int F;

  NegativeFieldInitializedInDefinition();
};
NegativeFieldInitializedInDefinition::NegativeFieldInitializedInDefinition() : F() {}

struct NegativeInClassInitialized {
  int F = 0;

  NegativeInClassInitialized() {}
};

struct NegativeInClassInitializedDefaulted {
  int F = 0;
  NegativeInClassInitializedDefaulted() = default;
};

struct NegativeConstructorDelegated {
  int F;

  NegativeConstructorDelegated(int F) : F(F) {}
  NegativeConstructorDelegated() : NegativeConstructorDelegated(0) {}
};

struct NegativeInitializedInBody {
  NegativeInitializedInBody() { I = 0; }
  int I;
};

struct A {};
template <class> class AA;
template <class T> class NegativeTemplateConstructor {
  NegativeTemplateConstructor(const AA<T> &, A) {}
  bool Bool{false};
  // CHECK-FIXES: bool Bool{false};
};

#define UNINITIALIZED_FIELD_IN_MACRO_BODY(FIELD) \
  struct UninitializedField##FIELD {             \
    UninitializedField##FIELD() {}               \
    int FIELD;                                   \
  };                                             \
// Ensure FIELD is not initialized since fixes inside of macros are disabled.
// CHECK-FIXES: int FIELD;

UNINITIALIZED_FIELD_IN_MACRO_BODY(F);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: F
UNINITIALIZED_FIELD_IN_MACRO_BODY(G);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: G

#define UNINITIALIZED_FIELD_IN_MACRO_ARGUMENT(ARGUMENT) \
  ARGUMENT

UNINITIALIZED_FIELD_IN_MACRO_ARGUMENT(struct UninitializedFieldInMacroArg {
  UninitializedFieldInMacroArg() {}
  int Field;
});
// CHECK-MESSAGES: :[[@LINE-3]]:3: warning: constructor does not initialize these fields: Field
// Ensure FIELD is not initialized since fixes inside of macros are disabled.
// CHECK-FIXES: int Field;

struct NegativeAggregateType {
  int X;
  int Y;
  int Z;
};

struct PositiveTrivialType {
  PositiveTrivialType() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F

  NegativeAggregateType F;
  // CHECK-FIXES: NegativeAggregateType F{};
};

struct NegativeNonTrivialType {
  PositiveTrivialType F;
};

static void PositiveUninitializedTrivialType() {
  NegativeAggregateType X;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: uninitialized record type: 'X'
  // CHECK-FIXES: NegativeAggregateType X{};

  NegativeAggregateType A[10]; // Don't warn because this isn't an object type.
}

static void NegativeInitializedTrivialType() {
  NegativeAggregateType X{};
  NegativeAggregateType Y = {};
  NegativeAggregateType Z = NegativeAggregateType();
  NegativeAggregateType A[10]{};
  NegativeAggregateType B[10] = {};
  int C; // No need to initialize this because we don't have a constructor.
  int D[8];
  NegativeAggregateType E = {0, 1, 2};
  NegativeAggregateType F({});
}

struct NonTrivialType {
  NonTrivialType() = default;
  NonTrivialType(const NonTrivialType &RHS) : X(RHS.X), Y(RHS.Y) {}

  int X;
  int Y;
};

static void PositiveNonTrivialTypeWithCopyConstructor() {
  NonTrivialType T;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: uninitialized record type: 'T'
  // CHECK-FIXES: NonTrivialType T{};

  NonTrivialType A[8];
  // Don't warn because this isn't an object type
}

struct ComplexNonTrivialType {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constructor does not initialize these fields: Y
  NegativeFieldInitialized X;
  int Y;
  // CHECK-FIXES: int Y{};
};

static void PositiveComplexNonTrivialType() {
  ComplexNonTrivialType T;
}

struct NegativeStaticMember {
  static NonTrivialType X;
  static NonTrivialType Y;
  static constexpr NonTrivialType Z{};
};

NonTrivialType NegativeStaticMember::X;
NonTrivialType NegativeStaticMember::Y{};

struct PositiveMultipleConstructors {
  PositiveMultipleConstructors() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A, B

  PositiveMultipleConstructors(int) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A, B

  PositiveMultipleConstructors(const PositiveMultipleConstructors &) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A, B

  // FIXME: The fix-its here collide providing an erroneous fix
  int A, B;
  // CHECK-FIXES: int A{}{}{}, B{}{}{};
};

typedef struct {
  int Member;
} CStyleStruct;

struct PositiveUninitializedBase : public NegativeAggregateType, CStyleStruct {
  PositiveUninitializedBase() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NegativeAggregateType, CStyleStruct
  // CHECK-FIXES: PositiveUninitializedBase() : NegativeAggregateType(), CStyleStruct() {}
};

struct PositiveUninitializedBaseOrdering : public NegativeAggregateType,
                                           public NonTrivialType {
  PositiveUninitializedBaseOrdering() : B() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NegativeAggregateType, NonTrivialType
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: constructor does not initialize these fields: A
  // CHECK-FIXES: PositiveUninitializedBaseOrdering() : NegativeAggregateType(), NonTrivialType(), B() {}

  // This is somewhat pathological with the base class initializer at the end...
  PositiveUninitializedBaseOrdering(int) : B(), NonTrivialType(), A() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NegativeAggregateType
  // CHECK-FIXES: PositiveUninitializedBaseOrdering(int) : B(), NegativeAggregateType(), NonTrivialType(), A() {}

  PositiveUninitializedBaseOrdering(float) : B(), NegativeAggregateType(), A() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NonTrivialType
  // CHECK-FIXES: PositiveUninitializedBaseOrdering(float) : B(), NegativeAggregateType(), NonTrivialType(), A() {}

  int A, B;
  // CHECK-FIXES: int A{}, B;
};

// We shouldn't need to initialize anything because PositiveUninitializedBase
// has a user-defined constructor.
struct NegativeUninitializedBase : public PositiveUninitializedBase {
  NegativeUninitializedBase() {}
};

struct InheritedAggregate : public NegativeAggregateType {
  int F;
};

static InheritedAggregate NegativeGlobal;

enum TestEnum {
  A,
  B,
  C
};

enum class TestScopedEnum {
  A,
  B,
  C
};

struct PositiveEnumType {
  PositiveEnumType() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: X, Y
  // No proposed fixes, as we don't know whether value initialization for these
  // enums really makes sense.

  TestEnum X;
  TestScopedEnum Y;
};

extern "C" {
struct NegativeCStruct {
  int X, Y, Z;
};

static void PositiveCStructVariable() {
  NegativeCStruct X;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: uninitialized record type: 'X'
  // CHECK-FIXES: NegativeCStruct X{};
}
}

static void NegativeStaticVariable() {
  static NegativeCStruct S;
  (void)S;
}

union NegativeUnionInClass {
  NegativeUnionInClass() {} // No message as a union can only initialize one member.
  int X = 0;
  float Y;
};

union PositiveUnion {
  PositiveUnion() : X() {} // No message as a union can only initialize one member.
  PositiveUnion(int) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: union constructor should initialize one of these fields: X, Y

  int X;
  // CHECK-FIXES: int X{};

  // Make sure we don't give Y an initializer.
  float Y;
  // CHECK-FIXES-NOT: float Y{};
};

struct PositiveAnonymousUnionAndStruct {
  PositiveAnonymousUnionAndStruct() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A, B, Y, Z, C, D, E, F, X

  union {
    int A;
    // CHECK-FIXES: int A{};
    short B;
  };

  struct {
    int Y;
    // CHECK-FIXES: int Y{};
    char *Z;
    // CHECK-FIXES: char *Z{};

    struct {
      short C;
      // CHECK-FIXES: short C{};
      double D;
      // CHECK-FIXES: double D{};
    };

    union {
      long E;
      // CHECK-FIXES: long E{};
      float F;
    };
  };
  int X;
  // CHECK-FIXES: int X{};
};

// This check results in a CXXConstructorDecl with no body.
struct NegativeDeletedConstructor : NegativeAggregateType {
  NegativeDeletedConstructor() = delete;

  Template<int> F;
};

// This pathological template fails to compile if actually instantiated. It
// results in the check seeing a null RecordDecl when examining the base class
// initializer list.
template <typename T>
class PositiveSelfInitialization : NegativeAggregateType
{
  PositiveSelfInitialization() : PositiveSelfInitialization() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NegativeAggregateType
  // CHECK-FIXES: PositiveSelfInitialization() : NegativeAggregateType(), PositiveSelfInitialization() {}
};

class PositiveIndirectMember {
  struct {
    int *A;
    // CHECK-FIXES: int *A{};
  };

  PositiveIndirectMember() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A
};

void Bug30487()
{
  NegativeInClassInitializedDefaulted s;
}

struct PositiveVirtualMethod {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constructor does not initialize these fields: F
  int F;
  // CHECK-FIXES: int F{};
  virtual int f() = 0;
};

struct PositiveVirtualDestructor {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constructor does not initialize these fields: F
  PositiveVirtualDestructor() = default;
  int F;
  // CHECK-FIXES: int F{};
  virtual ~PositiveVirtualDestructor() {}
};

struct PositiveVirtualBase : public virtual NegativeAggregateType {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constructor does not initialize these bases: NegativeAggregateType
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: constructor does not initialize these fields: F
  int F;
  // CHECK-FIXES: int F{};
};

template <typename T>
struct PositiveTemplateVirtualDestructor {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constructor does not initialize these fields: F
  T Val;
  int F;
  // CHECK-FIXES: int F{};
  virtual ~PositiveTemplateVirtualDestructor() = default;
};

template struct PositiveTemplateVirtualDestructor<int>;

#define UNINITIALIZED_FIELD_IN_MACRO_BODY_VIRTUAL(FIELD) \
  struct UninitializedFieldVirtual##FIELD {              \
    int FIELD;                                           \
    virtual ~UninitializedFieldVirtual##FIELD() {}       \
  };                                                     \
// Ensure FIELD is not initialized since fixes inside of macros are disabled.
// CHECK-FIXES: int FIELD;

UNINITIALIZED_FIELD_IN_MACRO_BODY_VIRTUAL(F);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: F
UNINITIALIZED_FIELD_IN_MACRO_BODY_VIRTUAL(G);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: G

struct NegativeEmpty {
};

static void NegativeEmptyVar() {
  NegativeEmpty e;
  (void)e;
}

struct NegativeEmptyMember {
  NegativeEmptyMember() {}
  NegativeEmpty e;
};

struct NegativeEmptyBase : NegativeEmpty {
  NegativeEmptyBase() {}
};

struct NegativeEmptyArrayMember {
  NegativeEmptyArrayMember() {}
  char e[0];
};

struct NegativeIncompleteArrayMember {
  NegativeIncompleteArrayMember() {}
  char e[];
};

template <typename T> class NoCrash {
  class B : public NoCrash {
    template <typename U> B(U u) {}
  };
};

struct PositiveBitfieldMember {
  PositiveBitfieldMember() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
  unsigned F : 5;
};

struct NegativeUnnamedBitfieldMember {
  NegativeUnnamedBitfieldMember() {}
  unsigned : 5;
};

struct NegativeInitializedBitfieldMembers {
  NegativeInitializedBitfieldMembers() : F(3) { G = 2; }
  unsigned F : 5;
  unsigned G : 5;
};
