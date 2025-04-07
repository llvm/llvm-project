// RUN: %clang -gkey-instructions -gno-column-info -x c++ %s -gmlt -S -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -gno-column-info -x c %s -gmlt -S -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

float g;
void a() {
// CHECK: %0 = load float, ptr @g{{.*}}, !dbg [[G1R3:!.*]]
// CHECK: %conv = fptosi float %0 to i32{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store i32 %conv, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
    int a = g;
// CHECK: ret{{.*}}, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R3]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 3)
// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
