// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ -std=c++17 %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-CXX

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-C

int g;
void a(int A, int B) {
// CHECK: entry:
// The load gets associated with the branch rather than the store.
// TODO: Associating it with the store may be a better choice.
// CHECK: %0 = load i32, ptr %A.addr{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: store i32 %0, ptr @g{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: switch i32 %0, label %{{.*}} [
// CHECK:   i32 0, label %sw.bb
// CHECK:   i32 1, label %sw.bb1
// CHECK: ], !dbg [[G2R1:!.*]]
    switch ((g = A)) {
// CHECK: br label %sw.epilog[[#]], !dbg [[G3R1:!.*]]
    case 0: break;
    case 1: {
// CHECK: sw.bb1:
// CHECK: %1 = load i32, ptr %B.addr{{.*}}, !dbg [[G4R2:!.*]]
// CHECK: switch i32 %1, label %{{.*}} [
// CHECK:   i32 0, label %sw.bb2
// CHECK: ], !dbg [[G4R1:!.*]]
    switch ((B)) {
        case 0: {
// Test that assignments in constant-folded switches don't go missing.
// CHECK-CXX: sw.bb2:
// CHECK-CXX: store i32 1, ptr %C{{.*}}, !dbg [[G5R1:!.*]]
#ifdef __cplusplus
            switch (const int C = 1; C) {
                case 0: break;
                case 1: break;
                default: break;
            }
#endif
        };
    }
    };
    }
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK-CXX: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK-CXX: [[RET]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
// CHECK-C: [[RET]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
