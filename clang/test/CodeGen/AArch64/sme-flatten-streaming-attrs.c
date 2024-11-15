// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -target-feature +sme %s -o - | FileCheck %s

// REQUIRES: aarch64-registered-target

extern void was_inlined(void);

#define __flatten  __attribute__((flatten))
void fn(void) { was_inlined(); }
void fn_streaming_compatible(void) __arm_streaming_compatible { was_inlined(); }
void fn_streaming(void) __arm_streaming { was_inlined(); }
__arm_locally_streaming void fn_locally_streaming(void) { was_inlined(); }
__arm_new("za") void fn_streaming_new_za(void) __arm_streaming { was_inlined(); }

__flatten
void caller(void) {
    fn();
    fn_streaming_compatible();
    fn_streaming();
    fn_locally_streaming();
    fn_streaming_new_za();
}
// CHECK-LABEL: void @caller()
//  CHECK-NEXT: entry:
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming
//  CHECK-NEXT:   call void @fn_locally_streaming
//  CHECK-NEXT:   call void @fn_streaming_new_za

__flatten void caller_streaming_compatible(void) __arm_streaming_compatible {
    fn();
    fn_streaming_compatible();
    fn_streaming();
    fn_locally_streaming();
    fn_streaming_new_za();
}
// CHECK-LABEL: void @caller_streaming_compatible()
//  CHECK-NEXT: entry:
//  CHECK-NEXT:   call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming
//  CHECK-NEXT:   call void @fn_locally_streaming
//  CHECK-NEXT:   call void @fn_streaming_new_za

__flatten void caller_streaming(void) __arm_streaming {
    fn();
    fn_streaming_compatible();
    fn_streaming();
    fn_locally_streaming();
    fn_streaming_new_za();
}
// CHECK-LABEL: void @caller_streaming()
//  CHECK-NEXT: entry:
//  CHECK-NEXT:   call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming_new_za

__flatten __arm_locally_streaming
void caller_locally_streaming(void) {
    fn();
    fn_streaming_compatible();
    fn_streaming();
    fn_locally_streaming();
    fn_streaming_new_za();
}
// CHECK-LABEL: void @caller_locally_streaming()
//  CHECK-NEXT: entry:
//  CHECK-NEXT:   call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming_new_za
