// RUN: %clang -gkey-instructions -gno-column-info -x c++ %s -gmlt -S -emit-llvm -o - -target x86_64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -gno-column-info -x c %s -gmlt -S -emit-llvm -o - -target x86_64-windows-msvc \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct { int *p; } Ptr;
Ptr getPtr();
void f() {
// CHECK: %call = call i64{{.*}}, !dbg [[G1R3:!.*]]
// CHECK: [[gep:%.*]] = getelementptr inbounds nuw %struct.Ptr, ptr %p, i32 0, i32 0
// CHECK: [[i2p:%.*]] = inttoptr i64 %call to ptr, !dbg [[G1R2:!.*]]
// CHECK: store ptr [[i2p]], ptr [[gep]], align 8, !dbg [[G1R1:!.*]]
    Ptr p = getPtr();
// CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R3]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 3)
// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
