

// RUN: %clang_cc1 -O0 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int foo(int * __indexable ptr, int idx) {
	ptr += idx;
	return *ptr;
}

int main() {
    int a;
    int * __indexable p = &a;

    return foo(p, -1);
}

// CHECK: define{{.*}} i32 @foo({{.*}})
// ...
// CHECK: %[[UGE_RES:[0-9]+]] = icmp uge i64 {{%.*}}, {{%.*}}, !annotation ![[ANNOT_NEW_IDX_GE_OLD:[0-9]+]]
// CHECK: br i1 %[[UGE_RES]], label %{{[a-z0-9]+}}, label %[[LABEL_TRAP:[a-z0-9]+]], !prof ![[PROFILE_METADATA:[0-9]+]], !annotation ![[ANNOT_NEW_IDX_GE_OLD]]
// CHECK: [[LABEL_TRAP]]
// CHECK-NEXT:   call void @llvm.ubsantrap{{.*}} !annotation ![[ANNOT_NEW_IDX_GE_OLD]]
// ...
// CHECK: define{{.*}} i32 @main() #0 {
// ...
// CHECK: ![[ANNOT_NEW_IDX_GE_OLD]] = !{!"bounds-safety-check-new-indexable-ptr-ge-old"}
