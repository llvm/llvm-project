// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

int g;
float a() {
  if (a)
    return g;
  return 1;
}

// CHECK: entry:
// CHECK: br i1 true{{.*}}, !dbg [[G1R1:!.*]]

// CHECK: if.then:
// CHECK-NEXT: %0 = load i32, ptr @g{{.*}}, !dbg [[G2R4:!.*]]
// CHECK-NEXT: %conv = sitofp i32 %0 to float{{.*}}, !dbg [[G2R3:!.*]]
// CHECK-NEXT: store float %conv, ptr %retval{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: br label %return{{.*}}, !dbg [[G2R1:!.*]]

// CHECK: if.end:
// CHECK-NEXT: store float 1.000000e+00, ptr %retval{{.*}}, !dbg [[G3R2:!.*]]
// CHECK-NEXT: br label %return, !dbg [[G3R1:!.*]]

// CHECK: return:
// CHECK-NEXT:  %1 = load float, ptr %retval{{.*}}, !dbg [[G4R2:!.*]]
// CHECK-NEXT:  ret float %1{{.*}}, !dbg [[G4R1:!.*]]

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R4]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 4)
// CHECK: [[G2R3]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 3)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
