// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple %itanium_abi_triple %s -o - | FileCheck %s
// Test the various accessibility flags in the debug info.
struct A {
  // CHECK: ![[A:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A",

  // CHECK-DAG: !DISubprogram(name: "pub_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void pub_default();
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "pub_default_static",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagStaticMember)
  static int pub_default_static;
};

// CHECK: !DIDerivedType(tag: DW_TAG_inheritance,{{.*}} baseType: ![[A]],{{.*}} flags: DIFlagPublic, extraData: i32 0)
class B : public A {
public:
  // CHECK-DAG: !DISubprogram(name: "pub",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPublic | DIFlagPrototyped,
  void pub();
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "public_static",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPublic | DIFlagStaticMember)
  static int public_static;

protected:
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "prot_using",{{.*}} line: [[@LINE+3]],{{.*}} flags: DIFlagProtected)
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "prot_typedef",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagProtected)
  typedef int prot_typedef;
  using prot_using = prot_typedef;
  prot_using prot_member;

  // CHECK: !DISubprogram(name: "prot",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagProtected | DIFlagPrototyped,
  void prot();

private:
  // CHECK: !DISubprogram(name: "priv_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void priv_default();
};

class C {
public:
  // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "D",{{.*}} flags: DIFlagPublic | DIFlagTypePassByValue,
  struct D {
  };
protected:
  // CHECK: !DICompositeType(tag: DW_TAG_union_type, name: "E",{{.*}} flags: DIFlagProtected | DIFlagTypePassByValue,
  union E {
  };
private:
  // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "J",{{.*}} flags: DIFlagTypePassByValue,
  struct J {
  };
public:
  D d;
  E e;
  J j;
};

struct F {
private:
  // CHECK: !DICompositeType(tag: DW_TAG_union_type, name: "G",{{.*}} flags: DIFlagPrivate | DIFlagTypePassByValue,
  union G {
  };
public:
  G g;
};

union H {
private:
  // CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "I",{{.*}} flags: DIFlagPrivate | DIFlagTypePassByValue,
  class I {
  };
public:
  I i;
};

union U {
  // CHECK-DAG: !DISubprogram(name: "union_pub_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void union_pub_default();
private:
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "union_priv",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrivate)
  int union_priv;
};

// CHECK: !DISubprogram(name: "free",
// CHECK-SAME:          flags: DIFlagPrototyped,
// CHECK-SAME:          spFlags: DISPFlagDefinition
void free() {}

U u;
A a;
B b;
C c;
F f;
H h;
