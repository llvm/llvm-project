// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name dummy > %t/dummy.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/mod.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Spurious > %t/spurious.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/dummy.rsp
// RUN: %clang @%t/mod.rsp
// RUN: %clang @%t/spurious.rsp
// RUN: %clang @%t/tu.rsp -frewrite-includes -w -E -o - | FileCheck %s

// CHECK: int bar();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include "test.h"{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 2 "{{.*[/\\]}}main.c"{{$}}
// CHECK-NEXT: # 1 "{{.*[/\\]}}test.h" 1{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include "dummy.h"{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 1 "{{.*[/\\]}}test.h"{{$}}
// CHECK-NEXT: #pragma clang module import dummy /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 2 "{{.*[/\\]}}test.h"{{$}}
// CHECK-NEXT: # 3 "{{.*[/\\]}}main.c" 2{{$}}
// CHECK-NEXT: int foo();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include "dummy.h"{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 4 "{{.*[/\\]}}main.c"{{$}}
// CHECK-NEXT: #pragma clang module import dummy /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 5 "{{.*[/\\]}}main.c"{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include <Spurious/Missing.h>{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 5 "{{.*[/\\]}}main.c"{{$}}
// CHECK-NEXT: # 1 "{{.*[/\\]}}Missing.h" 1{{$}}
// CHECK-NEXT: /* empty */
// CHECK-NEXT: # 6 "{{.*[/\\]}}main.c" 2{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include <Mod.h>{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 6 "{{.*[/\\]}}main.c"{{$}}
// CHECK-NEXT: #pragma clang module import Mod /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 7 "{{.*[/\\]}}main.c"{{$}}

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only -fmodules DIR/main.c -F DIR/frameworks -I DIR -fmodules-cache-path=DIR/module-cache",
    "file": "DIR/t.c"
  }
]

//--- dummy.h
extern int dummy;

//--- module.modulemap
module dummy { header "dummy.h" }
module Mod { header "Mod.h" }
//--- frameworks/Spurious.framework/Modules/module.modulemap
framework module Spurious {
  umbrella header "Spurious.h"
  module * { export * }
}
//--- frameworks/Spurious.framework/Headers/Spurious.h
#include <Mod.h>
//--- frameworks/Spurious.framework/Headers/Missing.h
/* empty */
//--- Mod.h
typedef int mod_int;
//--- test.h
#include "dummy.h"
//--- main.c
int bar();
#include "test.h"
int foo();
#include "dummy.h"
#include <Spurious/Missing.h>
#include <Mod.h>
