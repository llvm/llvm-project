
// RUN: %clang -gkey-instructions -x c++ %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

_Complex float ci;
void test() {
// CHECK: %ci.real = load float, ptr @ci{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: %ci.imag = load float, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G1R2]]
// CHECK: store float %ci.real, ptr %lc.realp{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: store float %ci.imag, ptr %lc.imagp{{.*}}, !dbg [[G1R1]]
  _Complex float lc = ci;

// CHECK: %ci.real1 = load float, ptr @ci{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: %ci.imag2 = load float, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G2R2]]
// CHECK: store float %ci.real1, ptr @ci{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: store float %ci.imag2, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G2R1]]
  ci = ci;

// CHECK: %add.r = fadd float %ci.real5, %ci.real3, !dbg [[G3R2:!.*]]
// CHECK: %add.i = fadd float %ci.imag6, %ci.imag4, !dbg [[G3R2]]
// CHECK: store float %add.r, ptr @ci{{.*}}, !dbg [[G3R1:!.*]]
// CHECK: store float %add.i, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G3R1]]
  ci += ci;

// CHECK: %add = fadd float %0, %1, !dbg [[G4R2:!.*]]
// CHECK: store float %add, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G4R1:!.*]]
  __imag ci = __imag ci + __imag ci;

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[RET:!.*]] = !DILocation({{.*}}, atomGroup: [[#]], atomRank: [[#]])
