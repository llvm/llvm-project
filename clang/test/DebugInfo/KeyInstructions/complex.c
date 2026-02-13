// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-CXX

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-C

_Complex float ci;
float f;
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

// CHECK: %sub.r = fsub float %ci.real9, %ci.real7, !dbg [[G4R2:!.*]]
// CHECK: %sub.i = fsub float %ci.imag10, %ci.imag8, !dbg [[G4R2]]
// CHECK: store float %sub.r, ptr @ci, align 4, !dbg [[G4R1:!.*]]
// CHECK: store float %sub.i, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G4R1]]
  ci -= ci;

// There's control flow introduced here to skip around nan and lib calls
// (which is ignored in the test as none of those insts need be key). This does
// make PHIs "backup" instructions, which is... odd. FIXME: Do we want to make
// the instructions producing the values in the PHI backups instead/too?
// CHECK: %real_mul_phi = phi float [ %mul_r, %entry ], [ %mul_r, %complex_mul_imag_nan ], [ %coerce.real, %complex_mul_libcall ], !dbg [[G5R2:!.*]]
// CHECK: %imag_mul_phi = phi float [ %mul_i, %entry ], [ %mul_i, %complex_mul_imag_nan ], [ %coerce.imag, %complex_mul_libcall ], !dbg [[G5R2]]
// CHECK: store float %real_mul_phi, ptr @ci, align 4, !dbg [[G5R1:!.*]]
// CHECK: store float %imag_mul_phi, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1), align 4, !dbg [[G5R1]]
  ci *= ci;

// div goes straight to lib call, which gets saved into a temp.
// CHECK: %coerce21.real = load float, ptr %coerce21.realp, align 4, !dbg [[G6R2:!.*]]
// CHECK: %coerce21.imag = load float, ptr %coerce21.imagp, align 4, !dbg [[G6R2]]
// CHECK: store float %coerce21.real, ptr @ci, align 4, !dbg [[G6R1:!.*]]
// CHECK: store float %coerce21.imag, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1), align 4, !dbg [[G6R1]]
  ci /= ci;

// CHECK: %add = fadd float %0, %1, !dbg [[G7R2:!.*]]
// CHECK: store float %add, ptr getelementptr inbounds nuw ({ float, float }, ptr @ci, i32 0, i32 1){{.*}}, !dbg [[G7R1:!.*]]
  __imag ci = __imag ci + __imag ci;

#ifndef __cplusplus
// CHECK-C: %2 = load float, ptr @f, align 4
// CHECK-C: %add.r24 = fadd float %2, %ci.real22, !dbg [[G8R2:!.*]]
// CHECK-C: store float %add.r24, ptr @f, align 4, !dbg [[G8R1:!.*]]
  f += ci;

// CHECK-C: %3 = load float, ptr @f, align 4
// CHECK-C: %sub.r27 = fsub float %3, %ci.real25, !dbg [[G9R2:!.*]]
// CHECK-C: store float %sub.r27, ptr @f, align 4, !dbg [[G9R1:!.*]]
  f -= ci;

// CHECK-C: %coerce32.real = load float, ptr %coerce32.realp, align 4, !dbg [[G10R2:!.*]]
// CHECK-C: store float %coerce32.real, ptr @f, align 4, !dbg [[G10R1:!.*]]
  f /= ci;

// CHECK-C: %mul.rl = fmul float %5, %ci.real33, !dbg [[G11R2:!.*]]
// CHECK-C: store float %mul.rl, ptr @f, align 4, !dbg [[G11R1:!.*]]
  f *= ci;
#endif
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
// CHECK: [[G5R2]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 2)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[G6R2]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 2)
// CHECK: [[G6R1]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
// CHECK: [[G7R2]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 2)
// CHECK: [[G7R1]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)
// CHECK-C: [[G8R2]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 2)
// CHECK-C: [[G8R1]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 1)
// CHECK-C: [[G9R2]] = !DILocation({{.*}}, atomGroup: 9, atomRank: 2)
// CHECK-C: [[G9R1]] = !DILocation({{.*}}, atomGroup: 9, atomRank: 1)
// CHECK-C: [[G10R2]] = !DILocation({{.*}}, atomGroup: 10, atomRank: 2)
// CHECK-C: [[G10R1]] = !DILocation({{.*}}, atomGroup: 10, atomRank: 1)
// CHECK-C: [[G11R2]] = !DILocation({{.*}}, atomGroup: 11, atomRank: 2)
// CHECK-C: [[G11R1]] = !DILocation({{.*}}, atomGroup: 11, atomRank: 1)
// CHECK-C: [[RET]] = !DILocation({{.*}}, atomGroup: 12, atomRank: 1)
// CHECK-CXX: [[RET]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 1)
