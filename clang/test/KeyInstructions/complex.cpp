// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

_Complex int ci;
void test() {
// CHECK: %ci.real = load i32, ptr @ci{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: %ci.imag = load i32, ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G1R2]]
// CHECK: store i32 %ci.real, ptr @ci{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: store i32 %ci.imag, ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G1R1]]
  ci = ci;
// CHECK: %add.r = add i32 %ci.real3, %ci.real1, !dbg [[G2R2:!.*]]
// CHECK: %add.i = add i32 %ci.imag4, %ci.imag2, !dbg [[G2R2]]
// CHECK: store i32 %add.r, ptr @ci{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: store i32 %add.i, ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G2R1]]
  ci += ci;
// CHECK: %add = add nsw i32 %0, %1, !dbg [[G3R2:!.*]]
// CHECK: store i32 %add, ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G3R1:!.*]]
  __imag ci = __imag ci + __imag ci;
// CHECK: ret void, !dbg [[G4R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
