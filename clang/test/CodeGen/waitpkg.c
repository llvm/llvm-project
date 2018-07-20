// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -target-feature +waitpkg -Wall -pedantic -o - | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple i386-unknown-unknown -emit-llvm -target-feature +waitpkg -Wall -pedantic -o - | FileCheck %s

#include <immintrin.h>

#include <stddef.h>
#include <stdint.h>

void test_umonitor(void *address) {
  //CHECK-LABEL: @test_umonitor
  //CHECK: call void @llvm.x86.umonitor(i8* %{{.*}})
  return _umonitor(address);
}

uint8_t test_umwait(uint32_t control, uint64_t counter) {
  //CHECK-LABEL: @test_umwait
  //CHECK: call i8 @llvm.x86.umwait(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return _umwait(control, counter);
}

uint8_t test_tpause(uint32_t control, uint64_t counter) {
  //CHECK-LABEL: @test_tpause
  //CHECK: call i8 @llvm.x86.tpause(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return _tpause(control, counter);
}
