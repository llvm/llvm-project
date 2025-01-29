// RUN: %clang_cc1 -ftarget-variant-availability-checks -fsyntax-only -Wunguarded-availability -fdiagnostics-parseable-fixits -triple x86_64-apple-macos10.14 -darwin-target-variant-triple x86_64-apple-ios12-macabi %s 2>&1 | FileCheck %s

__attribute__((availability(macOS, introduced=10.15))) __attribute__((availability(ios, introduced=13)))
int function(void);

int use() {
  function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.15, macCatalyst 13, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:14-[[@LINE-2]]:14}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}

#define API_AVAILABLE(x, y) __attribute__((availability(__API_AVAILABLE_PLATFORM_##x))) __attribute__((availability(__API_AVAILABLE_PLATFORM_##y)))
#define __API_AVAILABLE_PLATFORM_macos(x) macos,introduced=x
#define __API_AVAILABLE_PLATFORM_ios(x) ios,introduced=x
#define __API_AVAILABLE_PLATFORM_macCatalyst(x) macCatalyst,introduced=x

API_AVAILABLE(macos(10.15), ios(13))
@interface NewClass
@end

@interface OldButOfferFixit
@property(copy) NewClass *prop;
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:1-[[@LINE-2]]:1}:"API_AVAILABLE(macos(10.15), macCatalyst(13))\n"

@end
