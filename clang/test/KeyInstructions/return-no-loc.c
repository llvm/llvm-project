// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// The implicit return here get the line number of the closing brace; make it
// key to match existing behaviour.

int a() {
  if (a)
    return 1;
}

// CHECK: br i1 true{{.*}}, !dbg [[G1R1:!.*]]

// CHECK: if.then:
// CHECK-NEXT: store i32 1, ptr %retval{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: br label %if.end, !dbg [[G2R1:!.*]]

// CHECK: if.end:
// CHECK-NEXT: %0 = load i32, ptr %retval{{.*}}, !dbg [[G3R2:!.*]]
// CHECK-NEXT: ret i32{{.*}}, !dbg [[G3R1:!.*]]

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
