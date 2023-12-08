// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11 -emit-llvm -o - %s | FileCheck %s

void use_at_available(void) {
  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 1, i32 10, i32 12, i32 0)
  // CHECK-NEXT: icmp ne
  if (__builtin_available(macos 10.12, *))
    ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 1, i32 10, i32 12, i32 0)
  // CHECK-NEXT: icmp ne
  if (@available(macos 10.12, *))
    ;

  // CHECK: call i32 @__isPlatformVersionAtLeast(i32 1, i32 10, i32 12, i32 42)
  // CHECK-NEXT: icmp ne
  if (__builtin_available(ios 10, macos 10.12.42, *))
    ;

  // CHECK-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK: br i1 true
  if (__builtin_available(ios 10, *))
    ;

  // This check should be folded: our deployment target is 10.11.
  // CHECK-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK: br i1 true
  if (__builtin_available(macos 10.11, *))
    ;
}

// CHECK: declare i32 @__isPlatformVersionAtLeast(i32, i32, i32, i32)
