// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __builtin_va_list va_list;

static __inline__ __attribute__((__always_inline__)) __attribute__((__format__(printf, 3, 0)))
int vsnprintf(char* const __attribute__((pass_object_size(1))) dest, int size, const char* format, va_list ap)
        __attribute__((overloadable)) {
    return __builtin___vsnprintf_chk(dest, size, 0, __builtin_object_size(((dest)), (1)), format, ap);
}

void t(const char* fmt, ...) {
  va_list args;
  __builtin_va_start(args, fmt);
  const int size = 512;
  char message[size];
  vsnprintf(message, size, fmt, args);
}

// CHECK: cir.func private @__vsnprintf_chk

// CHECK: cir.func internal private @_ZL9vsnprintfPcU17pass_object_size1iPKcP13__va_list_tag

// Implicit size parameter in arg %1
//
// FIXME: tag the param with an attribute to designate the size information.
//
// CHECK: %1 = cir.alloca !u64i, cir.ptr <!u64i>, ["", init] {alignment = 8 : i64}

// CHECK: cir.store %arg1, %1 : !u64i, cir.ptr <!u64i>

// CHECK: %10 = cir.load %1 : cir.ptr <!u64i>, !u64i
// CHECK: %11 = cir.load %3 : cir.ptr <!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CHECK: %12 = cir.load %4 : cir.ptr <!cir.ptr<!ty_22struct2E__va_list_tag22>>, !cir.ptr<!ty_22struct2E__va_list_tag22>
// CHECK: %13 = cir.call @__vsnprintf_chk(%6, %8, %9, %10, %11, %12)
