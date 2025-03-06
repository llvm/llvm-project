// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s

// Include ctrl-flow to stop ret value store being elided.

int g;
int &f(int &r) {
  if (r)
// CHECK: if.then:
// CHECK-NEXT: %2 = load ptr, ptr %r.addr{{.*}}, !dbg [[G2R3:!.*]]
// CHECK-NEXT: store ptr %2, ptr %retval{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: br label %return, !dbg [[G2R1:!.*]]
    return r;
  return g;
}

// CHECK: [[G2R3]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 3)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
