// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=unused-types %s -o - | FileCheck %s --check-prefixes=UNUSED_TYPES

// Test that debug info nodes for function-local types are emitted in their
// lexical block scopes, including cases when types have forward declarations.

void test() {
#line 10
  {
    struct S { int a; };
    class C { int b; };
    S s;
#line 20
    {
      C c;
    }
  }

#line 30
  {
    typedef char Char;
    using Int = int;
    Char c;
#line 40
    {
      Int i;
    }
  }

#line 50
  {
    enum E { a, b, c };
    enum class T { aa, bb, cc };
    E e = E::a;
#line 60
    {
      T t = T::aa;
    }
  }

#line 70
  {
    union U { int i; char c; };
    U u = { 256 };
  }

  // Opaque complete declaration.
#line 80
  {
    enum OpaqueE : int;
    OpaqueE opaqueE{};
// With #include between forward declaration of the enum, and its definition,
// this code block should be split into two DILexicalBlocks.
#include "Inputs/noop.inc"
    enum OpaqueE : int { a };
  }

#line 90
  {
    enum NeverDefinedOpaqueE : int;
    NeverDefinedOpaqueE neverDefinedOpaqueE{};

    struct ForwardS;
    ForwardS *ptr;
    struct ForwardS {
      int a;
    };

    struct SWithForwardDecl;
    struct SWithForwardDecl {
      int a;
    };
    SWithForwardDecl sWithForwardDecl;
  }
}

// CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "test", {{.*}}, spFlags: DISPFlagDefinition, {{.*}}, retainedNodes: ![[RN:[0-9]+]]
// CHECK: ![[RN]] = !{![[STRUCT:[0-9]+]], ![[CLASS:[0-9]+]], ![[TYPEDEF:[0-9]+]], ![[USING:[0-9]+]], ![[ENUM:[0-9]+]], ![[ENUM_CLASS:[0-9]+]], ![[UNION:[0-9]+]], ![[OPAQUE_ENUM:[0-9]+]], ![[ND_ENUM:[0-9]+]], ![[FWD_STRUCT:[0-9]+]], ![[S_WITH_FWD:[0-9]+]]}

// CHECK: ![[STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: ![[LBSCOPE_1:[0-9]+]]
// CHECK: ![[LBSCOPE_1]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 10,
// CHECK: ![[CLASS]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", scope: ![[LBSCOPE_1]]
// CHECK: ![[TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Char", scope: ![[LBSCOPE_2:[0-9]+]]
// CHECK: ![[LBSCOPE_2]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 30,
// CHECK: ![[USING]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Int", scope: ![[LBSCOPE_2]]
// CHECK: ![[ENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", scope: ![[LBSCOPE_3:[0-9]+]]
// CHECK: ![[LBSCOPE_3]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 50,
// CHECK: ![[ENUM_CLASS]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "T", scope: ![[LBSCOPE_3]]
// CHECK: ![[UNION]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "U", scope: ![[LBSCOPE_4:[0-9]+]]
// CHECK: ![[LBSCOPE_4]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 70,
// CHECK: ![[OPAQUE_ENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "OpaqueE", scope: ![[LBSCOPE_5:[0-9]+]]
// CHECK: ![[LBSCOPE_5]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 80,
// CHECK: ![[ND_ENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "NeverDefinedOpaqueE", scope: ![[LBSCOPE_6:[0-9]+]]
// CHECK: ![[LBSCOPE_6]] = distinct !DILexicalBlock(scope: ![[SP]], {{.*}}, line: 90,
// CHECK: ![[FWD_STRUCT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "ForwardS", scope: ![[LBSCOPE_6]]
// CHECK-SAME:                               flags: DIFlagFwdDecl
// CHECK: ![[S_WITH_FWD]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SWithForwardDecl", scope: ![[LBSCOPE_6]]

// CHECK: !DILocalVariable(name: "s", scope: ![[LBSCOPE_1]]
// CHECK-SAME:                        type: ![[STRUCT:]]
// CHECK: !DILocalVariable(name: "c", scope: ![[LBSCOPE_11:[0-9]+]]
// CHECK-SAME:                        type: ![[CLASS]]
// CHECK: ![[LBSCOPE_11]] = distinct !DILexicalBlock(scope: ![[LBSCOPE_1]], {{.*}}, line: 20,
// CHECK: !DILocalVariable(name: "c", scope: ![[LBSCOPE_2]]
// CHECK-SAME:                        type: ![[TYPEDEF:[0-9]+]]
// CHECK: !DILocalVariable(name: "i", scope: ![[LBSCOPE_21:[0-9]+]]
// CHECK-SAME:                        type: ![[USING]]
// CHECK: ![[LBSCOPE_21]] = distinct !DILexicalBlock(scope: ![[LBSCOPE_2]], {{.*}}, line: 40,
//
// CHECK: !DILocalVariable(name: "e", scope: ![[LBSCOPE_3]]
// CHECK-SAME:                        type: ![[ENUM:[0-9]+]]
// CHECK: !DILocalVariable(name: "t", scope: ![[LBSCOPE_31:[0-9]+]]
// CHECK-SAME:                        type: ![[ENUM_CLASS]]
// CHECK: ![[LBSCOPE_31]] = distinct !DILexicalBlock(scope: ![[LBSCOPE_3]], {{.*}}, line: 60,
//
// CHECK: !DILocalVariable(name: "u", scope: ![[LBSCOPE_4]]
// CHECK-SAME:                        type: ![[UNION:[0-9]+]]

void test_unused() {
#line 80
  {
    struct X {};
    typedef int Y; // typedef doesn't go to retainedTypes.
    enum Z { z };
  }
}

// UNUSED_TYPES: !DISubprogram(name: "test_unused", {{.*}}, retainedNodes: ![[RETAINED_TYPES:[0-9]+]]
// UNUSED_TYPES: ![[RETAINED_TYPES]] = !{![[UNUSED_STRUCT:[0-9]+]], ![[UNUSED_ENUM:[0-9]+]]}
// UNUSED_TYPES: ![[UNUSED_STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: ![[UNUSED_LB:[0-9]+]]
// UNUSED_TYPES: ![[UNUSED_LB]] = distinct !DILexicalBlock({{.*}}, line: 80
// UNUSED_TYPES: ![[UNUSED_ENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Z", scope: ![[UNUSED_LB]]

void test_lambda() {
  auto t = []() { struct F {}; return F(); };
  auto v = t();
}

// TODO: <lambda()>::F should have its scope specified.
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "F", file
