// Make sure that __cpuidex in cpuid.h doesn't conflict with the MS
// compatibility built in by ensuring compilation succeeds:
// RUN: %clang_cc1 %s -ffreestanding -fms-extensions -fms-compatibility \
// RUN:  -fms-compatibility-version=19.00 -triple x86_64-pc-windows-msvc -emit-llvm -o -

typedef __SIZE_TYPE__ size_t;

#include <intrin.h>
#include <cpuid.h>

int cpuid_info[4];

void test_cpuidex(unsigned level, unsigned count) {
  __cpuidex(cpuid_info, level, count);
}
