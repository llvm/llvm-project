// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,NONE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-address -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=thread -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,TSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=memory -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,MSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-memory -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,MSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=hwaddress -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,HWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=kernel-hwaddress -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,HWASAN

// CHECK-LABEL: @test_address
// NONE: ret i1 false
// ASAN: call i1 @llvm.allow.sanitize.address()
// TSAN: ret i1 false
// MSAN: ret i1 false
// HWASAN: ret i1 false
_Bool test_address() {
  return __builtin_allow_sanitize_check("address");
}

// CHECK-LABEL: @test_kernel_address
// NONE: ret i1 false
// ASAN: call i1 @llvm.allow.sanitize.address()
// TSAN: ret i1 false
// MSAN: ret i1 false
// HWASAN: ret i1 false
_Bool test_kernel_address() {
  return __builtin_allow_sanitize_check("kernel-address");
}

// CHECK-LABEL: @test_thread
// NONE: ret i1 false
// ASAN: ret i1 false
// TSAN: call i1 @llvm.allow.sanitize.thread()
// MSAN: ret i1 false
// HWASAN: ret i1 false
_Bool test_thread() {
  return __builtin_allow_sanitize_check("thread");
}

// CHECK-LABEL: @test_memory
// NONE: ret i1 false
// ASAN: ret i1 false
// TSAN: ret i1 false
// MSAN: call i1 @llvm.allow.sanitize.memory()
// HWASAN: ret i1 false
_Bool test_memory() {
  return __builtin_allow_sanitize_check("memory");
}

// CHECK-LABEL: @test_kernel_memory
// NONE: ret i1 false
// ASAN: ret i1 false
// TSAN: ret i1 false
// MSAN: call i1 @llvm.allow.sanitize.memory()
// HWASAN: ret i1 false
_Bool test_kernel_memory() {
  return __builtin_allow_sanitize_check("kernel-memory");
}

// CHECK-LABEL: @test_hwaddress
// NONE: ret i1 false
// ASAN: ret i1 false
// TSAN: ret i1 false
// MSAN: ret i1 false
// HWASAN: call i1 @llvm.allow.sanitize.hwaddress()
_Bool test_hwaddress() {
  return __builtin_allow_sanitize_check("hwaddress");
}

// CHECK-LABEL: @test_kernel_hwaddress
// NONE: ret i1 false
// ASAN: ret i1 false
// TSAN: ret i1 false
// MSAN: ret i1 false
// HWASAN: call i1 @llvm.allow.sanitize.hwaddress()
_Bool test_kernel_hwaddress() {
  return __builtin_allow_sanitize_check("kernel-hwaddress");
}
