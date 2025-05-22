
// RUN: %clang -gkey-instructions -x c++ %s -gmlt -gno-column-info -S -emit-llvm -o - -ftrivial-auto-var-init=pattern \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -gno-column-info -S -emit-llvm -o - -ftrivial-auto-var-init=pattern \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// The implicit-check-not is important; we don't want the GEPs created for the
// store locations to be included in the atom group.

int g;
void a() {
// CHECK: call void @llvm.memcpy{{.*}}%A{{.*}}, !dbg [[G1R1:!.*]]
    int A[] = { 1, 2, 3 };

// CHECK: store i32 1, ptr %{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: store i32 2, ptr %{{.*}}, !dbg [[G2R1]]
// CHECK: %0 = load i32, ptr @g{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: store i32 %0, ptr %{{.*}}, !dbg [[G2R1]]
    int B[] = { 1, 2, g };

// CHECK: call void @llvm.memset{{.*}}%big{{.*}}, !dbg [[G3R1:!.*]]
// CHECK: store i8 97{{.*}}, !dbg [[G3R1]]
// CHECK: store i8 98{{.*}}, !dbg [[G3R1]]
// CHECK: store i8 99{{.*}}, !dbg [[G3R1]]
// CHECK: store i8 100{{.*}}, !dbg [[G3R1]]
    char big[65536] = { 'a', 'b', 'c', 'd' };

// CHECK: call void @llvm.memset{{.*}}%arr{{.*}}, !dbg [[G4R1:!.*]]
    char arr[] = { 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, };

// CHECK: store i8 -86, ptr %uninit{{.*}}, !dbg [[G5R1:!.*]], !annotation
    char uninit; // -ftrivial-auto-var-init=pattern
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
