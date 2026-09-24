// RUN: %clang_cc1 -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=constructor -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "A"{{.*}}DIFlagTypePassByValue
struct A {
} TestA;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "B"{{.*}}flags: DIFlagFwdDecl
struct B {
  B();
} TestB;

// CHECK-DAG: ![[C:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C"{{.*}}DIFlagTypePassByValue
struct C {
  C() {}
} TestC;

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "D"{{.*}}DIFlagTypePassByValue
struct D {
  D();
};
D::D() {}

// Test for constexpr constructor.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "E"{{.*}}DIFlagTypePassByValue
struct E {
  constexpr E(){};
} TestE;

// Restored by this revert: a constexpr constructor that is only declared keeps
// the class exempt from constructor homing. See Aliased below for a case where
// narrowing the exemption to defined constructors homes the type nowhere.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DeclaredConstexpr"{{.*}}DIFlagTypePassByValue
struct DeclaredConstexpr {
  constexpr DeclaredConstexpr();
} TestDeclaredConstexpr;

// A constructor of a class template specialization is only instantiated, and
// so only defined, where it is used. Nothing constructs Aliased<const int, int>
// here - it is only read, through the common initial sequence it shares with
// Aliased<int, int> - so its constructor is defined in no translation unit and
// nothing anywhere homes the type, even though the type must be complete to
// read the member. Constructing through the mutable alternative and reading
// through the const-qualified one is what
// absl::container_internal::map_slot_type does with std::pair.
//
// See https://timsong-cpp.github.io/cppwp/n3337/class.mem#19 for
// the rule that allows an object to be constructed by a constructor
// of one type and read via another type.
//
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Aliased<int, int>"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Aliased<const int, int>"{{.*}}DIFlagTypePassByValue
template <class A, class B> struct Aliased {
  A first;
  B second;
  Aliased(const A &a, const B &b) : first(a), second(b) {}
};
union AliasedSlot {
  Aliased<const int, int> value;
  Aliased<int, int> mutable_value;
  AliasedSlot() {}
  ~AliasedSlot() {}
} TestAliasedSlot;
int ReadAliasedSlot() {
  TestAliasedSlot.mutable_value = Aliased<int, int>(1, 2);
  return TestAliasedSlot.value.first;
}

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "ConstexprAliased<int, int>"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "ConstexprAliased<const int, int>"{{.*}}DIFlagTypePassByValue
template <class A, class B> struct ConstexprAliased {
  A first;
  B second;
  constexpr ConstexprAliased(const A &a, const B &b) : first(a), second(b) {}
};
union ConstexprAliasedSlot {
  ConstexprAliased<const int, int> value;
  ConstexprAliased<int, int> mutable_value;
  ConstexprAliasedSlot() {}
  ~ConstexprAliasedSlot() {}
} TestConstexprAliasedSlot;
int ReadConstexprAliasedSlot() {
  TestConstexprAliasedSlot.mutable_value = ConstexprAliased<int, int>(1, 2);
  return TestConstexprAliasedSlot.value.first;
}

// Defined out-of-line constexpr constructor should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "OutOfLineConstexpr"{{.*}}DIFlagTypePassByValue
struct OutOfLineConstexpr {
  constexpr OutOfLineConstexpr();
} TestOutOfLineConstexpr;
constexpr OutOfLineConstexpr::OutOfLineConstexpr() {}

// Defined delegating constructor where delegated constructor is not defined
// should not emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Delegating"{{.*}}flags: DIFlagFwdDecl
struct Delegating {
  Delegating() : Delegating(42) {}
  Delegating(int);
} TestDelegating;

// Defined out-of-line delegating constructor where delegated constructor is not
// defined should not emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "OutOfLineDelegating"{{.*}}flags: DIFlagFwdDecl
struct OutOfLineDelegating {
  OutOfLineDelegating();
  OutOfLineDelegating(int);
} TestOutOfLineDelegating;
OutOfLineDelegating::OutOfLineDelegating() : OutOfLineDelegating(42) {}

// Defined delegating constructor where delegated constructor is defined should
// emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToDefined"{{.*}}DIFlagTypePassByValue
struct DelegatingToDefined {
  DelegatingToDefined() : DelegatingToDefined(42) {}
  DelegatingToDefined(int) {}
} TestDelegatingToDefined;

// Defined delegating constructor where delegated constructor is defined out of
// line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingToOutOfLine {
  DelegatingToOutOfLine() : DelegatingToOutOfLine(42) {}
  DelegatingToOutOfLine(int);
} TestDelegatingToOutOfLine;
DelegatingToOutOfLine::DelegatingToOutOfLine(int) {}

// Defined out-of-line delegating constructor where delegated constructor is
// defined should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingOutOfLine {
  DelegatingOutOfLine();
  DelegatingOutOfLine(int) {}
} TestDelegatingOutOfLine;
DelegatingOutOfLine::DelegatingOutOfLine() : DelegatingOutOfLine(42) {}

// Defined out-of-line delegating constructor where delegated constructor is
// defined out-of-line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingOutOfLineToOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingOutOfLineToOutOfLine {
  DelegatingOutOfLineToOutOfLine();
  DelegatingOutOfLineToOutOfLine(int);
} TestDelegatingOutOfLineToOutOfLine;
DelegatingOutOfLineToOutOfLine::DelegatingOutOfLineToOutOfLine()
    : DelegatingOutOfLineToOutOfLine(42) {}
DelegatingOutOfLineToOutOfLine::DelegatingOutOfLineToOutOfLine(int) {}

// Delegating constructor to a copy constructor should not enable constructor
// homing, so it should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToCopyCtor"{{.*}}DIFlagTypePassByValue
struct DelegatingToCopyCtor {
  DelegatingToCopyCtor(const DelegatingToCopyCtor&) = default;
  DelegatingToCopyCtor(const DelegatingToCopyCtor& val, int)
      : DelegatingToCopyCtor(val) {}
};
void TestDelegatingToCopyCtor(DelegatingToCopyCtor) {}

// Delegating constructor to a move constructor should not enable constructor
// homing, so it should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingToMoveCtor"{{.*}}DIFlagTypePassByValue
struct DelegatingToMoveCtor {
  DelegatingToMoveCtor(const DelegatingToMoveCtor&) = default;
  DelegatingToMoveCtor(DelegatingToMoveCtor&&) = default;
  DelegatingToMoveCtor(DelegatingToMoveCtor&& val, int)
      : DelegatingToMoveCtor(static_cast<DelegatingToMoveCtor&&>(val)) {}
};
void TestDelegatingToMoveCtor(DelegatingToMoveCtor) {}

// Defined delegating constexpr constructor where delegated constructor is also
// defined should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingConstexpr"{{.*}}DIFlagTypePassByValue
struct DelegatingConstexpr {
  constexpr DelegatingConstexpr() : DelegatingConstexpr(42) {}
  constexpr DelegatingConstexpr(int) {}
} TestDelegatingConstexpr;

// Defined out-of-line delegating constexpr constructor where delegated
// constructor is also defined out-of-line should emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "DelegatingConstexprOutOfLine"{{.*}}DIFlagTypePassByValue
struct DelegatingConstexprOutOfLine {
  constexpr DelegatingConstexprOutOfLine();
  constexpr DelegatingConstexprOutOfLine(int);
} TestDelegatingConstexprOutOfLine;
constexpr DelegatingConstexprOutOfLine::DelegatingConstexprOutOfLine()
    : DelegatingConstexprOutOfLine(42) {}
constexpr DelegatingConstexprOutOfLine::DelegatingConstexprOutOfLine(int) {}

// Test that a standard layout type in a union emits full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInUnion"{{.*}}DIFlagTypePassByValue
struct SLInUnion {
  int x;
  SLInUnion(int);
};

union SLUnion {
  SLInUnion u;
};
void TestSLUnion(SLUnion) {}

// Test that all types and their bases/fields in a standard-layout union are
// emitted with full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "ParentSLBase"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "ChildSL"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "ParentSL"{{.*}}DIFlagTypePassByValue
struct ParentSLBase{
  ParentSLBase();
};
struct ChildSL {
  int b;
  ChildSL();
};
struct ParentSL : ParentSLBase {
  ChildSL f;
  ParentSL();
};
union FollowMembers {
  ParentSL a;
  int b;
};
void TestFollowMembers(FollowMembers) {}

// Test that a template has its debug info emitted when in a standard-layout
// union.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "TemplatedSL<int>"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "TemplatedSL<float>"{{.*}}DIFlagTypePassByValue
template <typename T>
struct TemplatedSL {
  T x;
  TemplatedSL(T);
};

union TemplatedUnion {
  TemplatedSL<int> a;
  TemplatedSL<float> b;
};
void TestTemplatedUnion(TemplatedUnion) {}

// Test that a standard layout type in a non-standard-layout union does not
// emit full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInNonSLUnion"{{.*}}flags: DIFlagFwdDecl
struct NonSLBase {
  int x;
};
struct NonSL : NonSLBase {
  int x;
  NonSL(int);
};

struct SLInNonSLUnion {
  int x;
  SLInNonSLUnion(int);
};

union NonSLUnion {
  SLInNonSLUnion s;
  NonSL n;
};
void TestNonSLUnion(NonSLUnion) {}

// Test that a type nested in a standard-layout union follows the same rules
// and emits full debug info.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "NestedSL"{{.*}}DIFlagTypePassByValue
union NestedUnion {
  struct NestedSL {
    int a;
    NestedSL(int);
  } n;
};
void TestNestedUnion(NestedUnion) {}

// Test that recursive type completion happens through arrays.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInArray"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInMultiArray"{{.*}}DIFlagTypePassByValue
struct SLInArray {
  int x;
  SLInArray(int);
};
struct SLInMultiArray {
  int y;
  SLInMultiArray(int);
};
union ArrayUnion {
  SLInArray arr[3];
  SLInMultiArray multi_arr[2][4];
  int raw;
};
void TestArrayUnion(ArrayUnion) {}

// Test that recursive type completion ignores cv-qualifiers.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLConst"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLVolatile"{{.*}}DIFlagTypePassByValue
struct SLConst {
  int x;
  SLConst(int);
};
struct SLVolatile {
  int y;
  SLVolatile(int);
};
union CVUnion {
  const SLConst c;
  volatile SLVolatile v;
  int raw;
};
void TestCVUnion(CVUnion) {}

// Test that recursive type completion happens through templated fields.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInGenericUnion"{{.*}}DIFlagTypePassByValue
template <typename T>
union GenericUnion {
  T val;
  int raw;
};
struct SLInGenericUnion {
  int x;
  SLInGenericUnion(int);
};
void TestGenericUnion(GenericUnion<SLInGenericUnion>) {}

// Test that recursive type completion happens for anonymous standard-layout
// unions.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLInAnonUnion"{{.*}}DIFlagTypePassByValue
struct SLInAnonUnion {
  int x;
  SLInAnonUnion(int);
};
struct EnclosingStruct {
  union {
    SLInAnonUnion a;
    int b;
  } u;
};
void TestEnclosingStruct(EnclosingStruct) {}

// Test that recursive type completion follows inheritence of typedefs.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLDerived"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "EmptyBase"{{.*}}DIFlagTypePassByValue
typedef struct EmptyBase {
  EmptyBase(int);
} EmptyBaseAlias;
struct SLDerived : EmptyBaseAlias {
  int y;
  SLDerived(int);
};
union TypedefDerivedUnion {
  SLDerived d;
  int raw;
};
void TestTypedefDerivedUnion(TypedefDerivedUnion) {}

// Test that recursive type completion follows multiple inheritence.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLMultipleDerived"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "EmptyBase1"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "EmptyBase2"{{.*}}DIFlagTypePassByValue
struct EmptyBase1 {
  EmptyBase1(int);
};
struct EmptyBase2 {
  EmptyBase2(int);
};
struct SLMultipleDerived : EmptyBase1, EmptyBase2 {
  int x;
  SLMultipleDerived(int);
};
union MultipleDerivedUnion {
  SLMultipleDerived d;
  int raw;
};
void TestMultipleDerivedUnion(MultipleDerivedUnion) {}

// Test that recursive type completion follows inheritence of non-empty bases.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "EmptyDerived"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLBase"{{.*}}DIFlagTypePassByValue
struct SLBase {
  int y;
  SLBase(int);
};
struct EmptyDerived : SLBase {
  EmptyDerived(int);
};
union EmptyDerivedUnion {
  EmptyDerived d;
  int raw;
};
void TestEmptyDerivedUnion(EmptyDerivedUnion) {}

// Test that recursive type completion does not follow types through pointers.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLPointerInUnion"{{.*}}flags: DIFlagFwdDecl
struct SLPointerInUnion {
  int x;
  SLPointerInUnion(int);
};

union SLUnionPointer {
  SLPointerInUnion *u;
};
void TestSLUnionPointer(SLUnionPointer) {}

// Test that recursive type completion does not follow through pointer or
// reference members.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLPointerAndReferenceMembers"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLPointerMember"{{.*}}flags: DIFlagFwdDecl
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "SLReferenceMember"{{.*}}flags: DIFlagFwdDecl
struct SLPointerMember {
  int x;
  SLPointerMember(int);
};

struct SLReferenceMember {
  int x;
  SLReferenceMember(int);
};

struct SLPointerAndReferenceMembers {
  SLPointerMember *a;
  SLReferenceMember &b;
};

union SLPointerAndReferenceMembersUnion {
  SLPointerAndReferenceMembers a;
};
void TestSLPointerAndReferenceMembers(SLPointerAndReferenceMembersUnion) {}

// Test for trivial constructor.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "F"{{.*}}DIFlagTypePassByValue
struct F {
  F() = default;
  F(int) {}
  int i;
} TestF;

// Test for trivial constructor.
// CHECK-DAG: ![[G:.*]] ={{.*}}!DICompositeType({{.*}}name: "G"{{.*}}DIFlagTypePassByValue
// CHECK-DAG: !DICompositeType({{.*}}scope: ![[G]], {{.*}}DIFlagTypePassByValue
struct G {
  G() : g_(0) {}
  struct {
    int g_;
  };
} TestG;

// Test for an aggregate class with an implicit non-trivial default constructor
// that is not instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "H",{{.*}}DIFlagTypePassByValue
struct H {
  B b;
};
void f(H h) {}

// Test for an aggregate class with an implicit non-trivial default constructor
// that is instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "J",{{.*}}DIFlagTypePassByValue
struct J {
  B b;
};
void f(decltype(J()) j) {}

// Test for a class with trivial default constructor that is not instantiated.
// CHECK-DAG: !DICompositeType({{.*}}name: "K",{{.*}}DIFlagTypePassByValue
class K {
  int i;
};
void f(K k) {}

// CHECK-DAG: !DICompositeType({{.*}}name: "DeletedCtors",{{.*}}DIFlagTypePassBy
struct NonTrivial {
  NonTrivial();
};
struct DeletedCtors {
  DeletedCtors() = delete;
  constexpr DeletedCtors(int) = delete;
  DeletedCtors(const DeletedCtors &) = default;
  void f1();
  NonTrivial t;
};

const NonTrivial &f(const DeletedCtors &D) {
  return D.t;
}

// Test that we don't use constructor homing on lambdas.
// CHECK-DAG: ![[L:.*]] ={{.*}}!DISubprogram({{.*}}name: "L"
// CHECK-DAG: !DICompositeType({{.*}}scope: ![[L]], {{.*}}DIFlagTypePassByValue
void L() {
  auto func = [&]() {};
}

// Check that types are being added to retained types list.
// CHECK-DAG: !DICompileUnit{{.*}}retainedTypes: ![[RETAINED:[0-9]+]]
// CHECK-DAG: ![[RETAINED]] = {{.*}}![[C]]


struct VTableAndCtor {
  virtual void f1();
  VTableAndCtor();
};

VTableAndCtor::VTableAndCtor() {
}

// ITANIUM-DAG: !DICompositeType({{.*}}name: "VTableAndCtor", {{.*}}flags: DIFlagFwdDecl

