// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s

// Cases to show some non-trivial types with flags combined with DIFlagNonTrivial and DIFlagTypePassByValue.

// CHECK-DAG: !DICompositeType({{.*}}, name: "Explicit",{{.*}}flags: DIFlagTypePassByValue | DIFlagNonTrivial
struct Explicit {
  explicit Explicit();
  int a;
} Explicit;

// CHECK-DAG: !DICompositeType({{.*}}, name: "Struct",{{.*}}flags: DIFlagTypePassByValue | DIFlagNonTrivial
struct Struct {
  Struct() {}
} Struct;

// CHECK-DAG: !DICompositeType({{.*}}, name: "Annotated",{{.*}}flags: DIFlagTypePassByValue | DIFlagNonTrivial
struct __attribute__((trivial_abi)) Annotated {
  Annotated() {};
} Annotated;


// Check a non-composite type
// CHECK-DAG: !DIGlobalVariable(name: "GlobalVar", {{.*}}type: {{.*}}, isLocal: false, isDefinition: true)
int GlobalVar = 0;

// Cases to test composite type's triviality

// CHECK-DAG: !DICompositeType({{.*}}, name: "Union",{{.*}}flags: {{.*}}DIFlagTrivial
union Union {
  int a;
} Union;

// CHECK-DAG: !DICompositeType({{.*}}, name: "Trivial",{{.*}}flags: {{.*}}DIFlagTrivial
struct Trivial {
  int i;
} Trivial;

// CHECK-DAG: !DICompositeType({{.*}}, name: "TrivialA",{{.*}}flags: {{.*}}DIFlagTrivial
struct TrivialA {
  TrivialA() = default;
} TrivialA;

// CHECK-DAG: !DICompositeType({{.*}}, name: "TrivialB",{{.*}}flags: {{.*}}DIFlagTrivial
struct TrivialB {
  int m;
  TrivialB(int x) { m = x; }
  TrivialB() = default;
} TrivialB;

// CHECK-DAG: !DICompositeType({{.*}}, name: "TrivialC",{{.*}}flags: {{.*}}DIFlagTrivial
struct TrivialC {
  struct Trivial x;
} TrivialC;

// CHECK-DAG: !DICompositeType({{.*}}, name: "TrivialD",{{.*}}flags: {{.*}}DIFlagTrivial
struct NT {
  NT() {};
};
struct TrivialD {
  static struct NT x; // Member is non-trivial but is static.
} TrivialD;


// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivial",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivial {
  NonTrivial() {}
} NonTrivial;

// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivialA",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivialA {
  ~NonTrivialA();
} NonTrivialA;

// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivialB",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivialB {
  struct NonTrivial x;
} NonTrivialB;

// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivialC",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivialC {
  virtual void f() {}
} NonTrivialC;

// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivialD",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivialD : NonTrivial {
} NonTrivialD;

// CHECK-DAG: !DICompositeType({{.*}}, name: "NonTrivialE",{{.*}}flags: {{.*}}DIFlagNonTrivial
struct NonTrivialE : Trivial, NonTrivial {
} NonTrivialE;
