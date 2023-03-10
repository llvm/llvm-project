// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -emit-llvm -triple hexagon-unknown-linux-musl %s -o - | FileCheck %s
#include <stdarg.h>

struct AAA {
  int x;
  int y;
  int z;
  int d;
};

// CHECK:   call void @llvm.va_start(ptr %arraydecay)
// CHECK:   %arraydecay1 = getelementptr inbounds [1 x %struct.__va_list_tag],
// ptr %ap, i32 0, i32 0
// CHECK:   br label %vaarg.maybe_reg

// CHECK: vaarg.maybe_reg:                                  ; preds = %entry
// CHECK:   %__current_saved_reg_area_pointer_p = getelementptr inbounds
// %struct.__va_list_tag, ptr %arraydecay2, i32 0, i32 0
// CHECK:   %__current_saved_reg_area_pointer = load ptr, ptr
// %__current_saved_reg_area_pointer_p
// CHECK:   %__saved_reg_area_end_pointer_p = getelementptr inbounds
// %struct.__va_list_tag, ptr %arraydecay2, i32 0, i32 1
// CHECK:   %__saved_reg_area_end_pointer = load ptr, ptr
// %__saved_reg_area_end_pointer_p
// CHECK:   %__new_saved_reg_area_pointer = getelementptr i8, ptr
// %__current_saved_reg_area_pointer, i32 4
// CHECK:   %0 = icmp sgt ptr %__new_saved_reg_area_pointer,
// %__saved_reg_area_end_pointer
// CHECK:   br i1 %0, label %vaarg.on_stack, label %vaarg.in_reg

// CHECK: vaarg.in_reg:                                     ; preds =
// %vaarg.maybe_reg
// CHECK:   store ptr %__new_saved_reg_area_pointer, ptr
// %__current_saved_reg_area_pointer_p
// CHECK:   br label %vaarg.end

// CHECK: vaarg.on_stack:                                   ; preds =
// %vaarg.maybe_reg
// CHECK:   %__overflow_area_pointer_p = getelementptr inbounds
// %struct.__va_list_tag, ptr %arraydecay2, i32 0, i32 2
// CHECK:   %__overflow_area_pointer = load ptr, ptr %__overflow_area_pointer_p
// CHECK:   %__overflow_area_pointer.next = getelementptr i8, ptr
// %__overflow_area_pointer, i32 4
// CHECK:   store ptr %__overflow_area_pointer.next, ptr
// %__overflow_area_pointer_p
// CHECK:   store ptr %__overflow_area_pointer.next, ptr
// %__current_saved_reg_area_pointer_p
// CHECK:   br label %vaarg.end

// CHECK: vaarg.end:                                        ; preds =
// %vaarg.on_stack, %vaarg.in_reg
// CHECK:   %vaarg.addr = phi ptr [ %__current_saved_reg_area_pointer, %vaarg.in_reg ], [ %__overflow_area_pointer, %vaarg.on_stack
// ]
// CHECK:   %1 = load i32, ptr %vaarg.addr

struct AAA aaa = {100, 200, 300, 400};

int foo(int xx, ...) {
  va_list ap;
  int d;
  int ret = 0;
  struct AAA bbb;
  va_start(ap, xx);
  d = va_arg(ap, int);
  ret += d;
  bbb = va_arg(ap, struct AAA);
  ret += bbb.d;
  d = va_arg(ap, int);
  ret += d;
  va_end(ap);
  return ret;
}

int main(void) {
  int x;
  x = foo(1, 2, aaa, 4);
  return x;
}
