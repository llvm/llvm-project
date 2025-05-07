

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

int main(int argc, char **argv) {
  int len;
  int *__single __counted_by(len) foo__single__counted_by;

// CHECK:  store i32 0, ptr {{%.*}}, align 4, !annotation ![[ANNOT_ZEROINIT:[0-9]+]]
// ...
// CHECK:  ![[ANNOT_ZEROINIT]] = !{!"bounds-safety-zero-init"}
}
