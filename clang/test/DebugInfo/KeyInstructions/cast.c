// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check that when a cast is a Key Instruction backup we add its operand
// instruction, if there is one, to the source atom too.

float g;
void a() {
// CHECK: %0 = load float, ptr @g{{.*}}, !dbg [[G1R3:!.*]]
// CHECK: %conv = fptosi float %0 to i32{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store i32 %conv, ptr %a{{.*}}, !dbg [[G1R1:!.*]]
    int a = g;
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R3]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 3)
// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
