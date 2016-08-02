// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

struct A
{
  // CHECK: !DISubprogram(name: "a", linkageName: "_ZN1A1aEiz"
  // CHECK-SAME:          line: [[@LINE+2]]
  // CHECK-SAME:          type: ![[ATY:[0-9]+]]
  void a(int c, ...) {}
  // CHECK: ![[ATY]] = !DISubroutineType(types: ![[AARGS:[0-9]+]])
  // We no longer use an explicit unspecified parameter. Instead we use a trailing null to mean the function is variadic.
  // CHECK: ![[AARGS]] = !{null, !{{[0-9]+}}, !{{[0-9]+}}, null}
};

  // CHECK: !DISubprogram(name: "b", linkageName: "_Z1biz"
  // CHECK-SAME:          line: [[@LINE+2]]
  // CHECK-SAME:          type: ![[BTY:[0-9]+]]
void b(int c, ...) {
  // CHECK: ![[BTY]] = !DISubroutineType(types: ![[BARGS:[0-9]+]])
  // CHECK: ![[BARGS]] = !{null, !{{[0-9]+}}, null}

  A a;

  // CHECK: !DILocalVariable(name: "fptr"
  // CHECK-SAME:             line: [[@LINE+2]]
  // CHECK-SAME:             type: ![[PST:[0-9]+]]
  void (*fptr)(int, ...) = b;
  // CHECK: ![[PST]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BTY]],
}
