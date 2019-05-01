// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -darwin-target-variant-triple x86_64-apple-ios13-macabi -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MAC %s
// RUN: %clang_cc1 -triple x86_64-apple-ios13-macabi -darwin-target-variant-triple x86_64-apple-macosx10.14 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,IOS %s

void use_at_available() {

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 1, i32 10, i32 15, i32 0)
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.15, *))
    ;

  // MAC: call i32 @__isPlatformOrVariantPlatformVersionAtLeast(i32 1, i32 10, i32 15, i32 0, i32 2, i32 13, i32 1, i32 0)
  // MAC-NEXT: icmp ne
  // IOS: call i32 @__isPlatformOrVariantPlatformVersionAtLeast(i32 2, i32 13, i32 1, i32 0, i32 1, i32 10, i32 15, i32 0)
  // IOS-NEXT: icmp ne
  if (@available(macos 10.15, macCatalyst 13.1, *))
   ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 2, i32 13, i32 1, i32 0)
  // CHECK-NEXT: icmp ne
  if (@available(ios 13.1, *))
   ;

  // These checks should be partially folded
  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 2, i32 13, i32 1, i32 0)
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.11, macCatalyst 13.1, *))
    ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 1, i32 10, i32 15, i32 0)
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.15, ios 11, *))
    ;

  // These checks should be folded: our deployment target is higher.
  // CHECK-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-NOT: call i32 @__isPlatformOrVariantPlatformVersionAtLeast
  // CHECK: br i1 true
  if (__builtin_available(macos 10.11, ios 11, *))
    ;

  // CHECK: br i1 true
  if (@available(ios 11, *))
    ;

  // CHECK: br i1 true
  if (@available(macos 10.11, *))
    ;
}

// CHECK: declare i32 @__isPlatformVersionAtLeast(i32, i32, i32, i32)
// CHECK: declare i32 @__isPlatformOrVariantPlatformVersionAtLeast(i32, i32, i32, i32, i32, i32, i32, i32)
