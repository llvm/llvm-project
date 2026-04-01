
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
// CHECK: %0 = alloca i8, i64 4{{.*}}, !dbg [[B1:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[B1:!.*]], !annotation
// CHECK: store ptr %0, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
    void *a = __builtin_alloca(4);

// CHECK: %1 = alloca i8, i64 4{{.*}}, !dbg [[B2:!.*]]
// CHECK: call void @llvm.memset{{.*}}, !dbg [[B2:!.*]], !annotation
// CHECK: store ptr %1, ptr %b{{.*}}, !dbg [[G2R1:!.*]]
    void *b = __builtin_alloca_with_align(4, 8);

// CHECK: %2 = load <4 x float>, ptr @mat{{.*}}, !dbg [[G3R2:!.*]]
// CHECK: call void @llvm.matrix.column.major.store.v4f32{{.*}}, !dbg [[B3:!.*]]
    __builtin_matrix_column_major_store(mat, f4, sizeof(float) * 2);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[B4:!.*]]
    __builtin_bzero(f4, sizeof(float) * 2);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[B5:!.*]]
    __builtin_bcopy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[B6:!.*]]
    __builtin_memcpy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[B7:!.*]]
    __builtin_mempcpy(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[B8:!.*]]
    __builtin_memcpy_inline(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memcpy{{.*}}, !dbg [[B9:!.*]]
    __builtin___memcpy_chk(f4, f8, sizeof(float) * 4, -1);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[B10:!.*]]
    __builtin___memmove_chk(f4, f8, sizeof(float) * 4, -1);

// CHECK: call void @llvm.memmove{{.*}}, !dbg [[B11:!.*]]
    __builtin_memmove(f4, f8, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[B12:!.*]]
    __builtin_memset(f4, 0, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[B13:!.*]]
    __builtin_memset_inline(f4, 0, sizeof(float) * 4);

// CHECK: call void @llvm.memset{{.*}}, !dbg [[B14:!.*]]
    __builtin___memset_chk(f4, 0, sizeof(float), -1);

// CHECK: %3 = load i32, ptr @v{{.*}}, !dbg [[G4R3:!.*]]
// CHECK-NEXT: %4 = trunc i32 %3 to i8, !dbg [[B15:!.*]]
// CHECK-NEXT: call void @llvm.memset{{.*}}, !dbg [[B15:!.*]]
    __builtin_memset(f4, v, sizeof(float) * 4);
// CHECK: ret{{.*}}, !dbg [[G5R1:!.*]]
}

// CHECK: [[B1]] = !DILocation(line: 0, scope: [[S1:!.*]], inlinedAt: [[I1:!.*]])
// CHECK: [[S1]] = distinct !DISubprogram(name: "__builtin_alloca"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I1]] = !DILocation(line: 19,
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[B2]] = !DILocation(line: 0, scope: [[S2:!.*]], inlinedAt: [[I2:!.*]])
// CHECK: [[S2]] = distinct !DISubprogram(name: "__builtin_alloca_with_align"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I2]] = !DILocation(line: 24,
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[B3]] = !DILocation(line: 0, scope: [[S3:!.*]], inlinedAt: [[I3:!.*]])
// CHECK: [[S3]] = distinct !DISubprogram(name: "__builtin_matrix_column_major_store"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I3]] = !DILocation(line: 28,
// CHECK: [[B4]] = !DILocation(line: 0, scope: [[S4:!.*]], inlinedAt: [[I4:!.*]])
// CHECK: [[S4]] = distinct !DISubprogram(name: "__builtin_bzero"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I4]] = !DILocation(line: 31,
// CHECK: [[B5]] = !DILocation(line: 0, scope: [[S5:!.*]], inlinedAt: [[I5:!.*]])
// CHECK: [[S5]] = distinct !DISubprogram(name: "__builtin_bcopy"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I5]] = !DILocation(line: 34,
// CHECK: [[B6]] = !DILocation(line: 0, scope: [[S6:!.*]], inlinedAt: [[I6:!.*]])
// CHECK: [[S6]] = distinct !DISubprogram(name: "__builtin_memcpy"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I6]] = !DILocation(line: 37,
// CHECK: [[B7]] = !DILocation(line: 0, scope: [[S7:!.*]], inlinedAt: [[I7:!.*]])
// CHECK: [[S7]] = distinct !DISubprogram(name: "__builtin_mempcpy"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I7]] = !DILocation(line: 40,
// CHECK: [[B8]] = !DILocation(line: 0, scope: [[S8:!.*]], inlinedAt: [[I8:!.*]])
// CHECK: [[S8]] = distinct !DISubprogram(name: "__builtin_memcpy_inline"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I8]] = !DILocation(line: 43,
// CHECK: [[B9]] = !DILocation(line: 0, scope: [[S9:!.*]], inlinedAt: [[I9:!.*]])
// CHECK: [[S9]] = distinct !DISubprogram(name: "__builtin___memcpy_chk"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I9]] = !DILocation(line: 46,
// CHECK: [[B10]] = !DILocation(line: 0, scope: [[S10:!.*]], inlinedAt: [[I10:!.*]])
// CHECK: [[S10]] = distinct !DISubprogram(name: "__builtin___memmove_chk"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I10]] = !DILocation(line: 49,
// CHECK: [[B11]] = !DILocation(line: 0, scope: [[S11:!.*]], inlinedAt: [[I11:!.*]])
// CHECK: [[S11]] = distinct !DISubprogram(name: "__builtin_memmove"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I11]] = !DILocation(line: 52,
// CHECK: [[B12]] = !DILocation(line: 0, scope: [[S12:!.*]], inlinedAt: [[I12:!.*]])
// CHECK: [[S12]] = distinct !DISubprogram(name: "__builtin_memset"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I12]] = !DILocation(line: 55,
// CHECK: [[B13]] = !DILocation(line: 0, scope: [[S13:!.*]], inlinedAt: [[I13:!.*]])
// CHECK: [[S13]] = distinct !DISubprogram(name: "__builtin_memset_inline"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I13]] = !DILocation(line: 58,
// CHECK: [[B14]] = !DILocation(line: 0, scope: [[S14:!.*]], inlinedAt: [[I14:!.*]])
// CHECK: [[S14]] = distinct !DISubprogram(name: "__builtin___memset_chk"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I14]] = !DILocation(line: 61,
// CHECK: [[G4R3]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 3)
// CHECK: [[B15]] = !DILocation(line: 0, scope: [[S12]], inlinedAt: [[I15:!.*]])
// CHECK: [[I15]] = !DILocation(line: 66,
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
