

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

struct Foo {
  int len;
  int *__single __counted_by(len) foo__single__counted_by;
};

int main(int argc, char **argv) {
  struct Foo f;
// CHECK:  [[LEN:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[STRUCT_F:%.*]], i32 0, i32 0
// CHECK:  store i32 0, ptr [[LEN]], align 8, !annotation ![[ANNOT_ZEROINIT:[0-9]+]]
// CHECK:  [[FOO_SINGLE_COUNTED_BY:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[STRUCT_F]], i32 0, i32 1
// CHECK:  store ptr null, ptr [[FOO_SINGLE_COUNTED_BY]], align 8, !annotation ![[ANNOT_ZEROINIT]]
// ...
// CHECK:  ![[ANNOT_ZEROINIT]] = !{!"bounds-safety-zero-init"}
}
