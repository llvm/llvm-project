

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

int main(int argc, char **argv) {
  int *__single foo__single[3];
// CHECK: [[TMP0:%.*]] = alloca [3 x ptr]
// CHECK: call void @llvm.memset.{{.*}}({{.*}} [[TMP0]], {{.*}} 0, {{.*}} 24{{.*}}){{.*}} !annotation ![[ANNOT_ZERO_INIT:[0-9]+]]
// ...
// CHECK: ![[ANNOT_ZERO_INIT]] = !{!"bounds-safety-zero-init"}
}
