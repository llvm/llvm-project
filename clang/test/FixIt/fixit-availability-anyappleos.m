// RUN: %clang_cc1 -triple x86_64-apple-macos26.0 -Wunguarded-availability -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// Test that the @available fix-it uses anyAppleOS when the availability attr
// was derived from anyAppleOS, and the platform-specific name when an explicit
// platform attr takes priority.

// Declaration with only anyAppleOS availability: fix-it should use anyAppleOS.
void func_anyappleos(void) __attribute__((availability(anyAppleOS, introduced=27.0)));

// Declaration with anyAppleOS + explicit macos: the macos attr takes priority,
// so the fix-it should use macOS and the macOS-specific version.
void func_macos(void)
    __attribute__((availability(anyAppleOS, introduced=27.0)))
    __attribute__((availability(macos, introduced=28.0)));

void test(void) {
  func_anyappleos();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(anyAppleOS 27.0, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:21-[[@LINE-2]]:21}:"\n  } else {\n      // Fallback on earlier versions\n  }"

  func_macos();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 28.0, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}
