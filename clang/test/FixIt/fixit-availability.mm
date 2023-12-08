// RUN: %clang_cc1 -fsyntax-only -Wunguarded-availability -fdiagnostics-parseable-fixits -triple x86_64-apple-darwin9 %s 2>&1 | FileCheck %s

__attribute__((availability(macos, introduced=10.12)))
int function(void);

void anotherFunction(int function);

int use() {
  function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:14-[[@LINE-2]]:14}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  int y = function(), x = 0;
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:29-[[@LINE-2]]:29}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  x += function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:19-[[@LINE-2]]:19}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  if (1) {
    x = function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:20-[[@LINE-2]]:20}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  }
  anotherFunction(function());
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:31-[[@LINE-2]]:31}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  if (function()) {
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE+1]]:4-[[@LINE+1]]:4}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  }
  while (function())
    // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
    // CHECK-NEXT: fix-it:{{.*}}:{[[@LINE+1]]:6-[[@LINE+1]]:6}:"\n  } else {\n      // Fallback on earlier versions\n  }"
    ;
  do
    function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n    } else {\n        // Fallback on earlier versions\n    }"
  while (1);
  for (int i = 0; i < 10; ++i)
    function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n    } else {\n        // Fallback on earlier versions\n    }"
  switch (x) {
  case 0:
    function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n    } else {\n        // Fallback on earlier versions\n    }"
  case 2:
    anotherFunction(1);
    function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n    } else {\n        // Fallback on earlier versions\n    }"
    break;
  default:
    function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n    } else {\n        // Fallback on earlier versions\n    }"
    break;
  }
  return function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:21-[[@LINE-2]]:21}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}

#define MYFUNCTION function

#define MACRO_ARGUMENT(X) X
#define MACRO_ARGUMENT_SEMI(X) X;
#define MACRO_ARGUMENT_2(X) if (1) X;

#define INNER_MACRO if (1) MACRO_ARGUMENT(function()); else ;

void useInMacros() {
  MYFUNCTION();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:"\n  } else {\n      // Fallback on earlier versions\n  }"

  MACRO_ARGUMENT_SEMI(function())
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:34-[[@LINE-2]]:34}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  MACRO_ARGUMENT(function());
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:30-[[@LINE-2]]:30}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  MACRO_ARGUMENT_2(function());
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:32-[[@LINE-2]]:32}:"\n  } else {\n      // Fallback on earlier versions\n  }"

  INNER_MACRO
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:14-[[@LINE-2]]:14}:"\n  } else {\n      // Fallback on earlier versions\n  }"
}

void wrapDeclStmtUses() {
  int x = 0, y = function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"if (@available(macOS 10.12, *)) {\n      "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE+13]]:22-[[@LINE+13]]:22}:"\n  } else {\n      // Fallback on earlier versions\n  }"
  {
    int z = function();
    if (z) {

    }
// CHECK: fix-it:{{.*}}:{[[@LINE-4]]:5-[[@LINE-4]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:6-[[@LINE-2]]:6}:"\n    } else {\n        // Fallback on earlier versions\n    }"
  }
  if (y)
    int z = function();
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:24-[[@LINE-2]]:24}:"\n    } else {\n        // Fallback on earlier versions\n    }"
  anotherFunction(y);
  anotherFunction(x);

  if (@available(macOS 10.1, *)) {
    int z = function();
    (void)z;
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:5-[[@LINE-2]]:5}:"if (@available(macOS 10.12, *)) {\n        "
// CHECK-NEXT: fix-it:{{.*}}:{[[@LINE-2]]:13-[[@LINE-2]]:13}:"\n    } else {\n        // Fallback on earlier versions\n    }"
    anotherFunction(x);
  }
}

#define API_AVAILABLE(X) __attribute__((availability(macos, introduced=10.12))) // dummy macro

API_AVAILABLE(macos(10.12))
@interface NewClass
@end

@interface OldButOfferFixit
@property(copy) NewClass *prop;
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:1-[[@LINE-2]]:1}:"API_AVAILABLE(macos(10.12))\n"

- (NewClass *)fixitMe;
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:22-[[@LINE-1]]:22}:" API_AVAILABLE(macos(10.12))"
@end

void oldButOfferFixitFn(NewClass *) {
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"API_AVAILABLE(macos(10.12))\n"
}

template<typename T>
struct OldButOfferFixitTag {
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:7-[[@LINE-1]]:7}:" API_AVAILABLE(macos(10.12))"
  NewClass *x;
};

// Avoid a fixit for declarations that already have an attribute:
__attribute__((availability(macos, introduced=10.11)))
@interface WithoutFixit
@property(copy) NewClass *prop;
// CHECK-NOT: API_AVAILABLE
@end
