// RUN: %clang -gkey-instructions -gno-column-info -x c++ %s -gmlt -S -emit-llvm -o - -target arm64-apple-ios11 \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -gno-column-info -x c %s -gmlt -S -emit-llvm -o - -target arm64-apple-ios11 \
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
// CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
