// RUN: %clang_cc1 -emit-llvm -fblocks -debug-info-kind=limited  -triple x86_64-apple-darwin14 -x objective-c < %s -o - | FileCheck %s
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr"
// CHECK-SAME:           baseType: ![[PTR:[0-9]+]]
// CHECK: ![[PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type,
// CHECK-SAME:                      baseType: ![[FNTYPE:[0-9]+]]
// CHECK: ![[FNTYPE]] = !DISubroutineType(types: ![[ARGS:[0-9]+]])
// CHECK: ![[ARGS]] = !{![[BOOL:.*]], ![[ID:.*]]}
#define nil ((void*) 0)
typedef signed char BOOL;
// CHECK: ![[BOOL]] = !DIDerivedType(tag: DW_TAG_typedef, name: "BOOL"
// CHECK-SAME:                              line: [[@LINE-2]]
// CHECK: ![[ID]] = !DIDerivedType(tag: DW_TAG_typedef, name: "id"

typedef BOOL (^SomeKindOfPredicate)(id obj);
int main()
{
  SomeKindOfPredicate p = ^BOOL(id obj) { return obj != nil; };
  // CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr",
  // CHECK-SAME:           line: [[@LINE-2]]
  // CHECK-SAME:           size: 64, offset: 128,
  return p(nil);
}
