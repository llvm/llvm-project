// RUN: %clang_cc1 -triple i386-unknown-unknown %s -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

int v;
enum e { MAX };

void foo (void)
{
  v = MAX;
}
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type,
// CHECK-SAME: baseType: ![[LONG:[0-9]+]]
// CHECK-SAME: elements: ![[ELTS:[0-9]+]]
// CHECK: ![[LONG]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
// CHECK: ![[ELTS]] = !{![[MAX:[0-9]+]]}
// CHECK: ![[MAX]] = !DIEnumerator(name: "MAX", value: 0)
