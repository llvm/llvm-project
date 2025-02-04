// Make sure that __cpuidex in cpuid.h doesn't conflict with the MS
// extensions built in by ensuring compilation succeeds:
// RUN: %clang_cc1 %s -ffreestanding -fms-extensions -fms-compatibility \
// RUN:  -fms-compatibility-version=19.00 -triple x86_64-pc-windows-msvc -emit-llvm -o -
// %clang_cc1 %s -ffreestanding -triple x86_64-w64-windows-gnu -fms-extensions -emit-llvm -o -
// RUN: %clang_cc1 %s -ffreestanding -fopenmp -fopenmp-is-target-device -aux-triple x86_64-unknown-linux-gnu

typedef __SIZE_TYPE__ size_t;

// We declare __cpuidex here as where the buitlin should be exposed (MSVC), the
// declaration is in <intrin.h>, but <intrin.h> is not available from all the
// targets that are being tested here.
void __cpuidex (int[4], int, int);

#include <cpuid.h>

int cpuid_info[4];

void test_cpuidex(unsigned level, unsigned count) {
  __cpuidex(cpuid_info, level, count);
}

