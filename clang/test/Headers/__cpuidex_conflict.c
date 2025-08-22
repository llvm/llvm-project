// Make sure that __cpuidex in cpuid.h doesn't conflict with the MS
// extensions built in by ensuring compilation succeeds:
// RUN: %clang_cc1 %s -DIS_STATIC="" -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 -triple x86_64-pc-windows-msvc -emit-llvm -o -
// RUN: %clang_cc1 %s -DIS_STATIC="" -ffreestanding -triple x86_64-w64-windows-gnu -fms-extensions -emit-llvm -o -

// Ensure that we do not run into conflicts when offloading.
// RUN: %clang_cc1 %s -DIS_STATIC=static -ffreestanding -fopenmp -fopenmp-is-target-device -aux-triple x86_64-unknown-linux-gnu
// RUN: %clang_cc1 -DIS_STATIC="" -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -aux-target-cpu x86-64 -fcuda-is-device -x cuda %s -o -

typedef __SIZE_TYPE__ size_t;

// We declare __cpuidex here as where the buitlin should be exposed (MSVC), the
// declaration is in <intrin.h>, but <intrin.h> is not available from all the
// targets that are being tested here.
IS_STATIC void __cpuidex (int[4], int, int);

#include <cpuid.h>

int cpuid_info[4];

void test_cpuidex(unsigned level, unsigned count) {
  __cpuidex(cpuid_info, level, count);
}
