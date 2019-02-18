// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
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
// CHECK: !DISubprogram(name: "-[I p1]",{{.*}} line: [[@LINE+4]],{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK: ![[DBG1]] = !DILocation(line: [[@LINE+3]],
// CHECK: !DISubprogram(name: "-[I setP1:]",{{.*}} line: [[@LINE+2]],{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK: ![[DBG2]] = !DILocation(line: [[@LINE+1]],
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
