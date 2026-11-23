// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple x86_64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple x86_64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct { int *p; } Ptr;
Ptr getPtr();
void f() {
// CHECK: %call = call i64{{.*}}, !dbg [[G1R3:!.*]]
// CHECK: [[gep:%.*]] = getelementptr inbounds nuw %struct.Ptr, ptr %p, i32 0, i32 0
// CHECK: [[i2p:%.*]] = inttoptr i64 %call to ptr, !dbg [[G1R2:!.*]]
// CHECK: store ptr [[i2p]], ptr [[gep]], align 8, !dbg [[G1R1:!.*]]
    Ptr p = getPtr();
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R3]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 3)
// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
