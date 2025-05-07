
// XFAIL: *
// BoundsSafety doesn't allow ended_by to reference a __bidi_indexable or __indexable pointer.
// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>
struct S {
    int *__ended_by(end) start;
    int *__bidi_indexable end;
};

int Foo(void) {
    int arr[10];
    struct S s = {arr, arr + 10};
    int *ptr = s.start;
    return 0;
}

// CHECK: [[STRUCT_TY:%.*]] = type { i32*, [[BIDI_TY:%.*]] }
// CHECK: [[BIDI_TY]] = type { i32*, i32*, i32* }

// CHECK-LABEL: @Foo
// CHECK:  getelementptr inbounds [[STRUCT_TY]], [[STRUCT_TY]]* [[S_ALLOC:%.*]], i32 0, i32 0
// ...
// CHECK:  [[S_START_ADDR:%.*]] = getelementptr inbounds [[STRUCT_TY]], [[STRUCT_TY]]* [[S_ALLOC:%.*]], i32 0, i32 0
// CHECK:  [[S_START:%.*]] = load i32*, i32** [[S_START_ADDR]], align 8
// CHECK:  [[PTR_PTR:%.*]] = getelementptr inbounds [[BIDI_TY]], [[BIDI_TY]]* [[PTR:%.*]], i32 0, i32 0
// CHECK:  store i32* [[S_START]], i32** [[PTR_PTR]], align 8
// CHECK:  [[S_END:%.*]] = getelementptr inbounds [[STRUCT_TY]], [[STRUCT_TY]]* [[S_ALLOC]], i32 0, i32 1
// ...
// CHECK:  call void @llvm.memcpy.p0i8.p0i8.i64{{.*}}
// CHECK:  [[TMP_END_ADDR:%.*]] = getelementptr inbounds [[BIDI_TY]], [[BIDI_TY]]* [[TEMP_END_LOC:%.*]], i32 0, i32 0
// CHECK:  [[TMP_END_PTR:%.*]] = load i32*, i32** [[TMP_END_ADDR]], align 8
// CHECK:  [[PTR_UPPER:%.*]] = getelementptr inbounds [[BIDI_TY]], [[BIDI_TY]]* [[PTR]], i32 0, i32 1
// CHECK:  store i32* [[TMP_END_PTR]], i32** [[PTR_UPPER]], align 8
// CHECK:  [[PTR_LOWER:%.*]] = getelementptr inbounds [[BIDI_TY]], [[BIDI_TY]]* [[PTR]], i32 0, i32 2
// CHECK:  store i32* [[S_START]], i32** [[PTR_LOWER]], align 8
