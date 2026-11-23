// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple arm64-apple-ios11 \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple arm64-apple-ios11 \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct {
  char a;
  int x;
} __attribute((packed)) S;

S getS();
void f() {
// CHECK: [[call:%.*]] = call i40{{.*}}getS{{.*}}, !dbg [[G1R2:!.*]]
// CHECK: store i40 [[call]], ptr %s, align 1, !dbg [[G1R1:!.*]]
    S s = getS();
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
