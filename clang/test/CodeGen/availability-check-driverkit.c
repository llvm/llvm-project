// RUN: %clang_cc1 -triple x86_64-apple-driverkit19.0 -emit-llvm -o - %s | FileCheck %s

void use_at_available() {
  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 10, i32 19, i32 1, i32 0)
  // CHECK-NEXT: icmp ne
  if (__builtin_available(driverkit 19.1, *))
    ;
}

// CHECK: declare i32 @__isPlatformVersionAtLeast(i32, i32, i32, i32)
