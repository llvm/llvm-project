// RUN: %clang_cc1 -fsyntax-only -Wunguarded-availability -fdiagnostics-parseable-fixits -triple x86_64-apple-ios13.1-macabi %s 2>&1 | FileCheck %s

__attribute__((availability(macCatalyst, introduced=13.2))) __attribute__((availability(ios, introduced=13.1)))
int function(void);

void anotherFunction(int function);

int use() {
  function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macCatalyst 13.2, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:14-[[@LINE-2]]:14}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}

#define API_AVAILABLE(X) __attribute__((availability(macCatalyst, introduced=13.2))) __attribute__((availability(ios, introduced=13.1))) // dummy macro

API_AVAILABLE(macos(10.12))
@interface NewClass
@end

@interface OldButOfferFixit
@property(copy) NewClass *prop;
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:1-[[@LINE-2]]:1}:"API_AVAILABLE(maccatalyst(13.2))\n"

@end
