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
  // CHECK: !DISubprogram(name: "prot",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagProtected | DIFlagPrototyped,
  void prot();
private:
  // CHECK: !DISubprogram(name: "priv_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void priv_default();
};

union U {
  // CHECK-DAG: !DISubprogram(name: "union_pub_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void union_pub_default();
private:
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "union_priv",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrivate)
  int union_priv;
};


// CHECK: !DISubprogram(name: "free",
// CHECK-SAME:          isDefinition: true
// CHECK-SAME:          flags: DIFlagPrototyped,
void free() {}

U u;
A a;
B b;
