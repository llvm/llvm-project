
// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ %s -debug-info-kind=line-tables-only -gno-column-info -emit-llvm -o - -ftrivial-auto-var-init=zero -fenable-matrix -disable-llvm-passes \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -gno-column-info -emit-llvm -o - -ftrivial-auto-var-init=zero -fenable-matrix -disable-llvm-passes \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef float m2x2 __attribute__((matrix_type(2, 2)));
m2x2 mat;
float f4[4];
float f8[8];
int v = 3;

void fun() {
// CHECK: %a = alloca ptr, align 8
// CHECK: %0 = alloca i8, i64 4{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[G1R1:!.*]], !annotation
// CHECK: store ptr %0, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
    void *a = __builtin_alloca(4);

// CHECK: %1 = alloca i8, i64 4{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[G2R1:!.*]], !annotation
// CHECK: store ptr %1, ptr %b{{.*}}, !dbg [[G2R1:!.*]]
    void *b = __builtin_alloca_with_align(4, 8);

// CHECK: %2 = load <4 x float>, ptr @mat{{.*}}, !dbg [[G3R2:!.*]]
// CHECK: call void @llvm.matrix.column.major.store.v4f32{{.*}}, !dbg [[G3R1:!.*]]
    __builtin_matrix_column_major_store(mat, f4, sizeof(float) * 2);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G4R1:!.*]]
    __builtin_bzero(f4, sizeof(float) * 2);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[G5R1:!.*]]
    __builtin_bcopy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G6R1:!.*]]
    __builtin_memcpy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G7R1:!.*]]
    __builtin_mempcpy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G8R1:!.*]]
    __builtin_memcpy_inline(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[G9R1:!.*]]
    __builtin___memcpy_chk(f4, f8, sizeof(float) * 4, -1);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[G10R1:!.*]]
    __builtin___memmove_chk(f4, f8, sizeof(float) * 4, -1);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[G11R1:!.*]]
    __builtin_memmove(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G12R1:!.*]]
    __builtin_memset(f4, 0, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G13R1:!.*]]
    __builtin_memset_inline(f4, 0, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G14R1:!.*]]
    __builtin___memset_chk(f4, 0, sizeof(float), -1);

// CHECK: %3 = load i32, ptr @v{{.*}}, !dbg [[G15R3:!.*]]
// CHECK-NEXT: %4 = trunc i32 %3 to i8, !dbg [[G15R2:!.*]]
// CHECK-NEXT: call void @llvm.memset{{.*}}, !dbg [[G15R1:!.*]]
    __builtin_memset(f4, v, sizeof(float) * 4);
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[G6R1]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
// CHECK: [[G7R1]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)
// CHECK: [[G8R1]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 1)
// CHECK: [[G9R1]] = !DILocation({{.*}}, atomGroup: 9, atomRank: 1)
// CHECK: [[G10R1]] = !DILocation({{.*}}, atomGroup: 10, atomRank: 1)
// CHECK: [[G11R1]] = !DILocation({{.*}}, atomGroup: 11, atomRank: 1)
// CHECK: [[G12R1]] = !DILocation({{.*}}, atomGroup: 12, atomRank: 1)
// CHECK: [[G13R1]] = !DILocation({{.*}}, atomGroup: 13, atomRank: 1)
// CHECK: [[G14R1]] = !DILocation({{.*}}, atomGroup: 14, atomRank: 1)
// CHECK: [[G15R3]] = !DILocation({{.*}}, atomGroup: 15, atomRank: 3)
// CHECK: [[G15R2]] = !DILocation({{.*}}, atomGroup: 15, atomRank: 2)
// CHECK: [[G15R1]] = !DILocation({{.*}}, atomGroup: 15, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 16, atomRank: 1)
