// RUN: %clang -xc++ -target x86_64-linux-gnu -emit-llvm -S -o - %s | FileCheck %s

struct Empty {};

struct Empty emptyvar;

void take_args(int a, ...) {
    // CHECK:      %overflow_arg_area = load ptr, ptr %overflow_arg_area_p, align 8
    // CHECK-NEXT: %overflow_arg_area.next = getelementptr i8, ptr %overflow_arg_area, i32 0
    // CHECK-NEXT: store ptr %overflow_arg_area.next, ptr %overflow_arg_area_p, align 8
    __builtin_va_list l;
    __builtin_va_start(l, a);
    emptyvar = __builtin_va_arg(l, struct Empty);
    __builtin_va_end(l);
}
