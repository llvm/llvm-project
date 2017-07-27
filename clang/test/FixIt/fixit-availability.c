// RUN: %clang_cc1 -fsyntax-only -Wunguarded-availability -fdiagnostics-parseable-fixits -triple x86_64-apple-darwin9 %s 2>&1 | FileCheck %s

__attribute__((availability(macos, introduced=10.12)))
int function(void);

void use() {
  function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (__builtin_available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:14-[[@LINE-2]]:14}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}

__attribute__((availability(macos, introduced=10.12)))
struct New { };

struct NoFixit {
  struct New field;
};
// CHECK-NOT: API_AVAILABLE
