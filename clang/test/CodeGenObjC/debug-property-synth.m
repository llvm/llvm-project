// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -g %s -o - | FileCheck %s
// rdar://problem/9468526
//
// Setting a breakpoint on a property should create breakpoints in
// synthesized getters/setters.
//
@interface I {
  int _p1;
}
// Test that the linetable entries for the synthesized getter and
// setter are correct.
//
// CHECK: define {{.*}}[I p1]
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DBG1:[0-9]+]]
//
// CHECK: define {{.*}}[I setP1:]
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DBG2:[0-9]+]]
//
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE+4]]] [local] [def] [-[I p1]]
// CHECK: [ DW_TAG_subprogram ] [line [[@LINE+3]]] [local] [def] [-[I setP1:]]
// CHECK: ![[DBG1]] = !MDLocation(line: [[@LINE+2]],
// CHECK: ![[DBG2]] = !MDLocation(line: [[@LINE+1]],
@property int p1;
@end

@implementation I
@synthesize p1 = _p1;
@end

int main() {
  I *myi;
  myi.p1 = 2;
  return myi.p1;
}
