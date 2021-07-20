// RUN: %clang_cc1 -triple x86_64-apple-ios13.1-macabi -emit-llvm -o - %s | FileCheck %s

void use_at_available() {
  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 2, i32 14, i32 0, i32 0)
  // CHECK-NEXT: icmp ne i32
  if (__builtin_available(ios 14, *))
    ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 2, i32 13, i32 2, i32 0)
  // CHECK-NEXT: icmp ne i32
  if (@available(macCatalyst 13.2, *))
    ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 2, i32 13, i32 2, i32 0)
  // CHECK-NEXT: icmp ne i32
  if (__builtin_available(macCatalyst 13.2, macos 10.15.2, *))
    ;
}
