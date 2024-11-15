// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-fuchsia -O2 -flto -ffat-lto-objects \
// RUN:          -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fvisibility=hidden  -emit-llvm -o - %s \
// RUN:   | FileCheck %s

// CHECK: llvm.embedded.object
// CHECK-SAME: section ".llvm.lto"

// CHECK-LABEL: define hidden void @foo
//      CHECK: entry:
// CHECK-NEXT:   %cmp14.not = icmp eq i64 %len, 0
// CHECK-NEXT:   br i1 %cmp14.not, label %for.end7, label %for.cond1.preheader.preheader
//      CHECK: for.cond1.preheader.preheader:                    ; preds = %entry
// CHECK-NEXT:   %arrayidx.1 = getelementptr inbounds nuw i8, ptr %ptr, i64 4
// CHECK-NEXT:   br label %for.cond1.preheader

// CHECK-NOT: @llvm.type.test

// The code below is a reduced case from https://github.com/llvm/llvm-project/issues/112053
#define __PRINTFLIKE(__fmt, __varargs) __attribute__((__format__(__printf__, __fmt, __varargs)))
typedef void func(void* arg, const char* fmt, ...) __PRINTFLIKE(2, 3);
typedef __SIZE_TYPE__ size_t;
typedef unsigned long uintptr_t;

extern "C"
void foo(const void* ptr, size_t len, long disp_addr,
                     func* printf_func, void* printf_arg) {
  uintptr_t address = (uintptr_t)ptr;
  size_t count;

  for (count = 0; count < len; count += 16) {
    union {
      unsigned int buf[4];
      unsigned char cbuf[16];
    } u;
    size_t s = 10;
    size_t i;

    for (i = 0; i < s / 4; i++) {
      u.buf[i] = ((const unsigned int*)address)[i];
      printf_func(printf_arg, "%08x ", static_cast<unsigned int>(u.buf[i]));
    }
  }
}

