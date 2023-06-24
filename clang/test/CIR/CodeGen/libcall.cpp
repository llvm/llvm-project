// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __builtin_va_list va_list;

static __inline__ __attribute__((__always_inline__)) __attribute__((__format__(printf, 3, 0)))
int vsnprintf(char* const __attribute__((pass_object_size(1))) dest, int size, const char* format, va_list ap)
        __attribute__((overloadable)) {
    return __builtin___vsnprintf_chk(dest, size, 0, 0, format, ap);
}

void t(const char* fmt, ...) {
  va_list args;
  __builtin_va_start(args, fmt);
  const int size = 512;
  char message[size];
  vsnprintf(message, size, fmt, args);
}

// CHECK: cir.func private @__vsnprintf_chk