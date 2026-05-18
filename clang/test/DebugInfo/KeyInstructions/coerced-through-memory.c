// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple aarch64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple aarch64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct {
  short a;
  int b;
  short c;
} S;

S getS(void);

void f() {
// CHECK: %call = call [2 x i64] {{.*}}getS{{.*}}(), !dbg [[G1R2:!.*]]
//// Note: The store to the tmp alloca isn't part of the atom.
// CHECK: store [2 x i64] %call, ptr %tmp.coerce, align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s, ptr align 8 %tmp.coerce, i64 12, i1 false), !dbg [[G1R1:!.*]]
  S s = getS();
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
