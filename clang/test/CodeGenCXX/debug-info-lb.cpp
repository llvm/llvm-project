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

// CHECK: !{{[0-9]+}} = !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: [[LBScope:![0-9]+]],
// CHECK: [[LBScope]] = distinct !DILexicalBlock(scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 5)

// CHECK: [[FuncScope:![0-9]+]] = distinct !DISubprogram(name: "foo",

// CHECK: !{{[0-9]+}} = !DIDerivedType(tag: DW_TAG_typedef, name: "Y", scope: [[LBScope]],
// CHECK: !{{[0-9]+}} = !DIGlobalVariable(name: "bar", scope: [[FuncScope]], file: !{{[0-9]+}}, line: 4
// CHECK: !{{[0-9]+}} = !DIGlobalVariable(name: "bar", scope: [[LBScope]], file: !{{[0-9]+}}, line: 8

