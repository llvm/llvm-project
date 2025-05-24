// RUN: %clang_cc1 -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s

struct B {
  float y;
};

class Cls {
  public:
    int x = 1;
    B b = {5.f};
};

void fun() {
  Cls c;
}

// CHECK: store i32 1, ptr %x{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: store float 5.000000e+00, ptr %y{{.*}}, !dbg [[G2R1:!.*]]

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
