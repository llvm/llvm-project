// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

void test__cpuid(int *info, int level) {
  __cpuid(info, level);
}
// CHECK-LABEL: define {{.*}} @test__cpuid(i32* %{{.*}}, i32 %{{.*}})
// CHECK: call { i32, i32, i32, i32 } asm "cpuid",
// CHECK-SAME:   "={ax},={bx},={cx},={dx},{ax},{cx},~{dirflag},~{fpsr},~{flags}"
// CHECK-SAME:   (i32 %{{.*}}, i32 0)
