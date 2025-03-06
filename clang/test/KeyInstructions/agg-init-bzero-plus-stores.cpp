// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

char g;
void a() {
// CHECK: _Z1av()
// CHECK: call void @llvm.memset{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: store i8 97{{.*}}, !dbg [[G1R1]]
// CHECK: store i8 98{{.*}}, !dbg [[G1R1]]
// CHECK: store i8 99{{.*}}, !dbg [[G1R1]]
// CHECK: store i8 100{{.*}}, !dbg [[G1R1]]
  char big[65536] = { 'a', 'b', 'c', 'd' };

// CHECK: call void @llvm.memset{{.*}}, !dbg [[G2R1:!.*]]
// CHECK: [[l:%.*]] = load i8{{.*}}, !dbg [[G2R2:!.*]]
// CHECK: store i8 [[l]]{{.*}}, !dbg [[G2R1C22:!.*]]
  char big2[65536] = { g };
// CHECK: ret void{{.*}}, !dbg [[G3R1:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G2R1C22]] = !DILocation({{.*}}column: 22{{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
