// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple aarch64-none-linux-android24 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __builtin_va_list va_list;

#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#define va_copy(dst, src)   __builtin_va_copy(dst, src)

// CHECK: [[VALISTTYPE:!.+va_list.*]] = !cir.struct<struct "{{.*}}__va_list

int average(int count, ...) {
// CHECK: cir.func @{{.*}}average{{.*}}(%arg0: !s32i loc({{.+}}), ...) -> !s32i
    va_list args, args_copy;
    va_start(args, count);
    // CHECK: cir.va.start %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>

    va_copy(args_copy, args);
    // CHECK: cir.va.copy %{{[0-9]+}} to %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>, !cir.ptr<[[VALISTTYPE]]>

    int sum = 0;
    for(int i = 0; i < count; i++) {
        sum += va_arg(args, int);
        // CHECK: %{{[0-9]+}} = cir.va.arg %{{[0-9]+}} : (!cir.ptr<[[VALISTTYPE]]>) -> !s32i
    }

    va_end(args);
    // CHECK: cir.va.end %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>

    return count > 0 ? sum / count : 0;
}

int test(void) {
  return average(5, 1, 2, 3, 4, 5);
  // CHECK: cir.call @{{.*}}average{{.*}}(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!s32i, !s32i, !s32i, !s32i, !s32i, !s32i) -> !s32i
}
