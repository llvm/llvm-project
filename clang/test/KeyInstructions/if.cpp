// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

int g;
void a(int A) {
// CHECK: entry:
// CHECK: %tobool = icmp ne i32 %0, 0{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: br i1 %tobool, label %if.then, label %if.end{{.*}}, !dbg [[G1R1:!.*]]
    if (A)
        ;
// The assignment in the if currently gets a distinct source atom group.
// FIXME: Is that the right choice?
// CHECK: if.end:
// CHECK: %1 = load i32, ptr %A.addr{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: store i32 %1, ptr @g{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: %tobool1 = icmp ne i32 %1, 0{{.*}}, !dbg [[G3R2:!.*]]
// CHECK: br i1 %tobool1, label %if.then2, label %if.end3{{.*}}, !dbg [[G3R1:!.*]]
    if ((g = A))
        ;
// The assignment in the if currently gets a distinct source atom group.
// FIXME: Is that the right choice?
// CHECK: if.end3:
// CHECK: %2 = load i32, ptr %A.addr{{.*}}, !dbg [[G4R2:!.*]]
// CHECK: store i32 %2, ptr %B{{.*}}, !dbg [[G4R1:!.*]]
// CHECK: %tobool4 = icmp ne i32 %3, 0{{.*}}, !dbg [[G5R2:!.*]]
// CHECK: br i1 %tobool4, label %if.then5, label %if.end6{{.*}}, !dbg [[G5R1:!.*]]
    if (int B = A; B)
        ;
// CHECK: ret{{.*}}, !dbg [[G6R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[G5R2]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 2)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[G6R1]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
