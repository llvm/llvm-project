// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s

class Cls {
  int x = 1;
};

void fun() {
  Cls c;
}

// CHECK: store i32 1, ptr %x{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
