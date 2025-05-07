

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

struct Foo {
  int *__single foo__single;
  int *__indexable foo__indexable;
  int *__bidi_indexable foo__bidi_indexable;
  int len;
  int *__single __counted_by(len) foo__single__counted_by;
};

int main(int argc, char **argv) {
  struct Foo f;
// CHECK: [[FOO_SINGLE:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[FOO_F:%.*]], i32 0, i32 0
// CHECK: store ptr null, ptr [[FOO_SINGLE]]{{.*}} !annotation ![[ANNOT_ZERO_INIT:[0-9]+]]
// CHECK: [[FOO_INDEXABLE:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[FOO_F]], i32 0, i32 1
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[FOO_INDEXABLE]], i8 0, i64 16, i1 false){{.*}} !annotation ![[ANNOT_ZERO_INIT:[0-9]+]]
// CHECK: [[FOO_BIDI_INDEX:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[FOO_F]], i32 0, i32 2
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[FOO_BIDI_INDEX]], i8 0, i64 24, i1 false){{.*}} !annotation ![[ANNOT_ZERO_INIT]]
// CHECK: getelementptr inbounds nuw %struct.Foo, ptr [[FOO_F]], i32 0, i32 3
// CHECK: [[FOO_SINGLE_COUNTED_BY:%.*]] = getelementptr inbounds nuw %struct.Foo, ptr [[FOO_F]], i32 0, i32 4
// CHECK: store ptr null, ptr [[FOO_SINGLE_COUNTED_BY]], align 8{{.*}} !annotation ![[ANNOT_ZERO_INIT]]
// ...
// CHECK: ![[ANNOT_ZERO_INIT]] = !{!"bounds-safety-zero-init"}
}
