// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -darwin-target-variant-triple x86_64-apple-ios12-macabi -emit-llvm -o - %s | FileCheck %s

// XFAIL: *
void use_at_available() {

  // CHECK: call i32 @__isOSVersionAtLeast(i32 10, i32 15, i32 0)
  // CHECK-NEXT: call i32 @__isTargetPlatformNative()
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: select
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.15, *))
    ;

  // CHECK: call i32 @__isOSVersionAtLeast(i32 10, i32 15, i32 0)
  // CHECK-NEXT: call i32 @__isTargetVariantOSVersionAtLeast(i32 13, i32 0, i32 0)
  // CHECK-NEXT: call i32 @__isTargetPlatformNative()
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: select
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.15, iosmac 13, *))
   ;

  // CHECK: call i32 @__isTargetVariantOSVersionAtLeast(i32 13, i32 0, i32 0)
  // CHECK-NEXT: call i32 @__isTargetPlatformNative()
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: select
  // CHECK-NEXT: icmp ne
  if (@available(ios 13, *))
   ;

  // This check should be folded: our deployment target is 10.11.
  // CHECK-NOT: call i32 @__isOSVersionAtLeast
  // CHECK-NOT: call i32 @__isTargetVariantOSVersionAtLeast
  // CHECK: br i1 true
  if (__builtin_available(macos 10.11, ios 11, *))
    ;
}

// CHECK: declare i32 @__isOSVersionAtLeast(i32, i32, i32)
// CHECK: declare i32 @__isTargetPlatformNative
// CHECK: declare i32 @__isTargetVariantOSVersionAtLeast(i32, i32, i32)
