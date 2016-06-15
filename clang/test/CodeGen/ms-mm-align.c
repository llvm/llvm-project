// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -target-feature +sse \
// RUN:         -triple i686--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s -check-prefix CHECK

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

void capture_ptr(int* i);
void test_mm_align16(int p) {
  _MM_ALIGN16 int i;
  capture_ptr(&i);
}

// CHECK: alloca i32, align 16
