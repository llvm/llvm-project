
// XFAIL: *

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null

#include <ptrcheck.h>

void Foo(int *__counted_by(len) buf, int len) {}

unsigned long get_len(void *__bidi_indexable ptr);
unsigned long trap_if_bigger_than_max(unsigned long len);


int Test() {
    int arr[10];
    Foo(arr, trap_if_bigger_than_max(get_len(arr)));
    return 0;
}

// CHECK-LABEL: @Test
// ...
// CHECK:   [[RET_GET_LEN:%.*]] = call i64 @get_len
// CHECK:   [[RET_TRAP_IF:%.*]] = call i64 @trap_if_bigger_than_max(i64 [[RET_GET_LEN]]), !annotation [[ANNOT_CONV_TO_COUNT:![0-9]+]]
// ...
// CHECK:   [[COND_LE_UB:%.*]] = icmp ule i64 {{%.*}}, {{%.*}}, !annotation [[ANNOT_LE_UB:![0-9]+]]
// CHECK:   br i1 [[COND_LE_UB]], label %[[LABEL_CONT:.*]], label %[[LABEL_TRAP:.*]], !annotation [[ANNOT_LE_UB]]

// CHECK: [[LABEL_TRAP]]:
// CHECK:   call void @llvm.ubsantrap{{.*}}, !annotation [[ANNOT_LE_UB]]
// CHECK:   unreachable

// 'RET_TRAP_IF' - the result of call 'trap_if_bigger_than_max' is used in count check here and later as argument for function call to @Foo,
// instead of being re-evaluated.
// CHECK: [[LABEL_CONT]]:
// CHECK:   [[COND_CONV_TO_COUNT:%.*]] = icmp ule i64 [[RET_TRAP_IF]], {{.*}}, !annotation [[ANNOT_CONV_TO_COUNT]]
// CHECK:   br i1 [[COND_CONV_TO_COUNT]], label %[[LABEL_CONT12:.*]], label %[[LABEL_TRAP11:.*]], !annotation [[ANNOT_CONV_TO_COUNT]]

// CHECK: [[LABEL_TRAP11]]:                                           ; preds = %[[LABEL_CONT]]
// CHECK:   call void @llvm.ubsantrap{{.*}}, !annotation [[ANNOT_CONV_TO_COUNT]]
// CHECK:   unreachable

// CHECK: [[LABEL_CONT12]]:                                           ; preds = %[[LABEL_CONT]]
// CHECK:   [[ARG_CONV:%.*]] = trunc i64 [[RET_TRAP_IF]] to i32
// CHECK:   call void @Foo(i32* {{%.*}}, i32 [[ARG_CONV]])
// CHECK:   ret i32 0

// CHECK: [[ANNOT_CONV_TO_COUNT]] = !{!"bounds-safety-check-conversion-to-count"}
// CHECK: [[ANNOT_LE_UB]] = !{!"bounds-safety-check-ptr-le-upper-bound"}
