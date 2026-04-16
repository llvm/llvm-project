// RUN: %clang_cc1 -triple x86_64-apple-macosx27.0 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-MACOS %s
// RUN: %clang_cc1 -triple arm64-apple-ios27.0 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-IOS %s
// RUN: %clang_cc1 -triple aarch64-linux-android27 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-ANDROID %s

void test_anyappleos_older_than_deployment(void) {
  // Deployment target is 27.0, so anyAppleOS 26.0 should fold to true.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_anyappleos_older_than_deployment
  // CHECK-MACOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-MACOS: br i1 true
  // CHECK-IOS-LABEL: define{{.*}} void @test_anyappleos_older_than_deployment
  // CHECK-IOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-IOS: br i1 true
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_anyappleos_older_than_deployment
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(anyAppleOS 26.0, *))
    ;
}

void test_anyappleos_equal_to_deployment(void) {
  // Deployment target is 27.0, so anyAppleOS 27.0 should fold to true.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_anyappleos_equal_to_deployment
  // CHECK-MACOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-MACOS: br i1 true
  // CHECK-IOS-LABEL: define{{.*}} void @test_anyappleos_equal_to_deployment
  // CHECK-IOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-IOS: br i1 true
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_anyappleos_equal_to_deployment
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(anyAppleOS 27.0, *))
    ;
}

void test_anyappleos_newer_than_deployment(void) {
  // Deployment target is 27.0, so anyAppleOS 28.0 requires runtime check.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_anyappleos_newer_than_deployment
  // CHECK-MACOS: call i32 @__isPlatformVersionAtLeast(i32 1, i32 28, i32 0, i32 0)
  // CHECK-MACOS-NEXT: icmp ne
  // CHECK-IOS-LABEL: define{{.*}} void @test_anyappleos_newer_than_deployment
  // CHECK-IOS: call i32 @__isPlatformVersionAtLeast(i32 2, i32 28, i32 0, i32 0)
  // CHECK-IOS-NEXT: icmp ne
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_anyappleos_newer_than_deployment
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(anyAppleOS 28.0, *))
    ;
}
void test_ios_check_on_macos(void) {
  // On macOS, checking for iOS should fold to true (different OS).
  // On iOS, checking for iOS 28.0 with deployment 27.0 requires runtime check.
  // On Android, checking for any Apple OS should fold to true.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_ios_check_on_macos
  // CHECK-MACOS: br i1 true
  // CHECK-IOS-LABEL: define{{.*}} void @test_ios_check_on_macos
  // CHECK-IOS: call i32 @__isPlatformVersionAtLeast(i32 2, i32 28, i32 0, i32 0)
  // CHECK-IOS-NEXT: icmp ne
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_ios_check_on_macos
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(ios 28.0, *))
    ;
}

void test_macos_check_on_ios(void) {
  // On macOS, checking for macOS 28.0 with deployment 27.0 requires runtime check.
  // On iOS, checking for macOS should fold to true (different OS).
  // On Android, checking for any Apple OS should fold to true.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_macos_check_on_ios
  // CHECK-MACOS: call i32 @__isPlatformVersionAtLeast(i32 1, i32 28, i32 0, i32 0)
  // CHECK-MACOS-NEXT: icmp ne
  // CHECK-IOS-LABEL: define{{.*}} void @test_macos_check_on_ios
  // CHECK-IOS: br i1 true
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_macos_check_on_ios
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(macos 28.0, *))
    ;
}

void test_priority(void) {
  // Platform-specific checks take priority over anyAppleOS.

  // On macOS: macos 28.0 applies (platform-specific), requires runtime check since deployment is 27.0.
  // On iOS: anyAppleOS 26.0 applies, folds to true since deployment is 27.0.
  // On Android: Non-Apple OS, folds to true.
  // CHECK-MACOS-LABEL: define{{.*}} void @test_priority
  // CHECK-MACOS: call i32 @__isPlatformVersionAtLeast(i32 1, i32 28, i32 0, i32 0)
  // CHECK-MACOS-NEXT: icmp ne
  // CHECK-IOS-LABEL: define{{.*}} void @test_priority
  // CHECK-IOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-IOS: br i1 true
  // CHECK-ANDROID-LABEL: define{{.*}} void @test_priority
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(anyAppleOS 26.0, macos 28.0, *))
    ;

  // Order of checks shouldn't matter; same behavior as above.
  // CHECK-MACOS: call i32 @__isPlatformVersionAtLeast(i32 1, i32 28, i32 0, i32 0)
  // CHECK-MACOS-NEXT: icmp ne
  // CHECK-IOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-IOS: br i1 true
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(macos 28.0, anyAppleOS 26.0, *))
    ;

  // On macOS: macos 26.0 applies (platform-specific), folds to true since deployment is 27.0.
  // On iOS: anyAppleOS 28.0 applies, requires runtime check since deployment is 27.0.
  // On Android: Non-Apple OS, folds to true.
  // CHECK-MACOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-MACOS: br i1 true
  // CHECK-IOS: call i32 @__isPlatformVersionAtLeast(i32 2, i32 28, i32 0, i32 0)
  // CHECK-IOS-NEXT: icmp ne
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(anyAppleOS 28.0, macos 26.0, *))
    ;

  // Order of checks shouldn't matter; same behavior as above.
  // CHECK-MACOS-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-MACOS: br i1 true
  // CHECK-IOS: call i32 @__isPlatformVersionAtLeast(i32 2, i32 28, i32 0, i32 0)
  // CHECK-IOS-NEXT: icmp ne
  // CHECK-ANDROID-NOT: call i32 @__isPlatformVersionAtLeast
  // CHECK-ANDROID: br i1 true
  if (__builtin_available(macos 26.0, anyAppleOS 28.0, *))
    ;
}
