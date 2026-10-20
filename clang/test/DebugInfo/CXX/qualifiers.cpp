// RUN: %clang_cc1 -std=c++11 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s
// Test (r)value and CVR qualifiers on C++11 non-static member functions.
class A {
public:
  // CHECK: !DISubprogram(name: "l",
  // CHECK-SAME:          line: [[@LINE+4]]
  // CHECK-SAME:          type: ![[PLSR:[0-9]+]]
  // CHECK-SAME:          flags: DIFlagPublic | DIFlagPrototyped | DIFlagLValueReference,
  // CHECK: ![[PLSR]] = !DISubroutineType(flags: DIFlagLValueReference, types: ![[ARGS:[0-9]+]])
  void l() const &;
  // CHECK: ![[ARGS]] = !{null, ![[THIS:[0-9]+]]}
  // CHECK: ![[THIS]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[CONST_A:[0-9]+]]
  // CHECK: ![[CONST_A]] = !DIDerivedType(tag: DW_TAG_const_type
  // CHECK: !DISubprogram(name: "r"
  // CHECK-SAME:          line: [[@LINE+4]]
  // CHECK-SAME:          type: ![[PRSR:[0-9]+]]
  // CHECK-SAME:          flags: DIFlagPublic | DIFlagPrototyped | DIFlagRValueReference,
  // CHECK: ![[PRSR]] = !DISubroutineType(flags: DIFlagRValueReference, types: ![[ARGS]])
  void r() const &&;
};

void g() {
  A a;
  // The type of pl is "void (A::*)() const &".
  // CHECK: !DILocalVariable(name: "pl",
  // CHECK-SAME:             line: [[@LINE+3]]
  // CHECK-SAME:             type: ![[PL:[0-9]+]]
  // CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: ![[PLSR]]
  auto pl = &A::l;

  // CHECK: !DILocalVariable(name: "pr",
  // CHECK-SAME:             line: [[@LINE+3]]
  // CHECK-SAME:             type: ![[PR:[0-9]+]]
  // CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: ![[PRSR]]
  auto pr = &A::r;
}
