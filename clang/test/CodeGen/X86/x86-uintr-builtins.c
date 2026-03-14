// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +uintr -emit-llvm -o - | FileCheck %s

#include <x86gprintrin.h>

void test_clui(void) {
// CHECK-LABEL: @test_clui
// CHECK: call void @llvm.x86.clui()
// CHECK: ret
  _clui();
}

void test_stui(void) {
// CHECK-LABEL: @test_stui
// CHECK: call void @llvm.x86.stui()
// CHECK: ret
  _stui();
}

unsigned char test_testui(void) {
// CHECK-LABEL: @test_testui
// CHECK: %[[TMP0:.+]] = call i8 @llvm.x86.testui()
// CHECK: ret i8 %[[TMP0]]
  return _testui();
}

void test_senduipi(unsigned long long a) {
// CHECK-LABEL: @test_senduipi
// CHECK: call void @llvm.x86.senduipi(i64 %{{.+}})
// CHECK: ret
  _senduipi(a);
}
