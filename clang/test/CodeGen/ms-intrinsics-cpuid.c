// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=X64

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

void test__cpuid(int *info, int level) {
  __cpuid(info, level);
}
// X86-LABEL: define {{.*}} @test__cpuid(i32* noundef %{{.*}}, i32 noundef %{{.*}})
// X86: call { i32, i32, i32, i32 } asm "cpuid",
// X86-SAME:   "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"
// X86-SAME:   (i32 %{{.*}}, i32 0)

// X64-LABEL: define {{.*}} @test__cpuid(i32* noundef %{{.*}}, i32 noundef %{{.*}})
// X64: call { i32, i32, i32, i32 } asm "xchg$(q$) $(%rbx{{.*}}$){{.*}}cpuid{{.*}}xchg$(q$) $(%rbx{{.*}}$)",
// X64-SAME:   "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"
// X64-SAME:   (i32 %{{.*}}, i32 0)
