
// RUN: %clang_cc1 -x c++ -gkey-instructions %s -debug-info-kind=line-tables-only -gno-column-info -emit-llvm -o - -ftrivial-auto-var-init=pattern \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -x c -gkey-instructions %s -debug-info-kind=line-tables-only -gno-column-info -emit-llvm -o - -ftrivial-auto-var-init=pattern \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// The implicit-check-not is important; we don't want the GEPs created for the
// store locations to be included in the atom group.

int g;
void a() {
// CHECK: call void @llvm.memcpy{{.*}}%A{{.*}}, !dbg [[G1R1:!.*]]
    int A[] = { 1, 2, 3 };

// CHECK:      call void @llvm.memcpy{{.*}}%B{{.*}}, !dbg [[G2R1:!.*]]
// CHECK-NEXT: store i32 1, ptr %B{{.*}}, !dbg [[G2R1:!.*]]
// CHECK-NEXT: %arrayinit.element = getelementptr {{.*}}, ptr %B, {{.*}} 1, !dbg [[B_LINE:!.*]]
// CHECK-NEXT: store i32 2, ptr %arrayinit.element{{.*}}, !dbg [[G2R1]]
// CHECK-NEXT: %arrayinit.element1 = getelementptr {{.*}}, ptr %B, {{.*}} 2, !dbg [[B_LINE]]
// CHECK-NEXT: %0 = load i32, ptr @g{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: store i32 %0, ptr %arrayinit.element1{{.*}}, !dbg [[G2R1]]
    int B[] = { 1, 2, g };

// CHECK:      call void @llvm.memset{{.*}}%big{{.*}} !dbg [[G3R1:!.*]]
// CHECK-NEXT: %1 = getelementptr {{.*}}, ptr %big, {{.*}} 0, {{.*}} 0, !dbg [[big_LINE:!.*]]
// CHECK-NEXT: store i8 97, ptr %1{{.*}}, !dbg [[G3R1]]
// CHECK-NEXT: %2 = getelementptr {{.*}}, ptr %big, {{.*}} 0, {{.*}} 1, !dbg [[big_LINE]]
// CHECK-NEXT: store i8 98, ptr %2{{.*}}, !dbg [[G3R1]]
// CHECK-NEXT: %3 = getelementptr {{.*}}, ptr %big, {{.*}} 0, {{.*}} 2, !dbg [[big_LINE]]
// CHECK-NEXT: store i8 99, ptr %3{{.*}}, !dbg [[G3R1]]
// CHECK-NEXT: %4 = getelementptr {{.*}}, ptr %big, {{.*}} 0, {{.*}} 3, !dbg [[big_LINE]]
// CHECK: store i8 100, ptr %4{{.*}} !dbg [[G3R1]]
    char big[65536] = { 'a', 'b', 'c', 'd' };

// CHECK: call void @llvm.memset{{.*}}%arr{{.*}}, !dbg [[G4R1:!.*]]
    char arr[] = { 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, };

// CHECK: store i8 -86, ptr %uninit{{.*}}, !dbg [[G5R1:!.*]], !annotation
    char uninit; // -ftrivial-auto-var-init=pattern
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[B_LINE]] = !DILocation(line: 23, scope: ![[#]])
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[big_LINE]] = !DILocation(line: 34, scope: ![[#]])
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
