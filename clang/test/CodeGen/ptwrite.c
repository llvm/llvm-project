// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +ptwrite -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s --check-prefix=X86 --check-prefix=X86_64
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +ptwrite -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s --check-prefix=X86

#include <x86intrin.h>

#include <stdint.h>

void test_ptwrite32(uint32_t value) {
  //X86-LABEL: @test_ptwrite32
  //X86: call void @llvm.x86.ptwrite32(i32 %{{.*}})
  _ptwrite32(value);
}

#ifdef __x86_64__

void test_ptwrite64(uint64_t value) {
  //X86_64-LABEL: @test_ptwrite64
  //X86_64: call void @llvm.x86.ptwrite64(i64 %{{.*}})
  _ptwrite64(value);
}

#endif /* __x86_64__ */
