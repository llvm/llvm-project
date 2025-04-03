
// RUN: %clang -gkey-instructions %s -gmlt -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct { int a, b, c; } Struct;
void fun(Struct a) {
// CHECK: call void @llvm.memcpy{{.*}}, !dbg  [[G1R1:!.*]]
  Struct b = a;
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)

