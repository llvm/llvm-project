// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

void foo() {
  static int bar = 1;
  {
    struct X {};
    typedef char Y;
    static int bar = 0;
    // The following basic block is intended, in order to check the case where
    // types "X", "Y" are defined in a different scope than where they are used.
    // They should have the scope they are defined at as their parent scope.
    {
      X a;
      Y b;
    }
  }
}

// CHECK: !{{[0-9]+}} = distinct !DIGlobalVariable(name: "bar", scope: [[FSCOPE:![0-9]+]]
// CHECK: [[FSCOPE]] = distinct !DISubprogram(name: "foo"
// CHECK: !{{[0-9]+}} = distinct !DIGlobalVariable(name: "bar", scope: [[LBSCOPE:![0-9]+]]
// CHECK: [[LBSCOPE]] = distinct !DILexicalBlock(scope: [[FSCOPE]]
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "a", scope: [[LBSCOPE2:![0-9]+]], {{.*}} type: [[STRUCT:![0-9]+]]
// CHECK: [[LBSCOPE2]] = distinct !DILexicalBlock(scope: [[LBSCOPE]]
// CHECK: [[STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: [[LBSCOPE]]
// CHECK: !{{[0-9]+}} = !DILocalVariable(name: "b", scope: [[LBSCOPE2]], {{.*}} type: [[TYPEDEF:![0-9]+]]
// CHECK: [[TYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Y", scope: [[LBSCOPE]]
