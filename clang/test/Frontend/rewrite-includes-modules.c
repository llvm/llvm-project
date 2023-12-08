// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'extern int dummy;' > %t/dummy.h
// RUN: echo 'module dummy { header "dummy.h" }' > %t/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -I%t -E -frewrite-includes -o - | FileCheck %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c %s -I%t -E -frewrite-includes -o - | FileCheck %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ %s -I%t -E -frewrite-includes -o - | FileCheck %s

int bar();
#include "dummy.h"
int foo();
#include "dummy.h"

// CHECK: int bar();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include "dummy.h"{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 10 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: #pragma clang module import dummy /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 11 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: int foo();{{$}}
// CHECK-NEXT: #if 0 /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: #include "dummy.h"{{$}}
// CHECK-NEXT: #endif /* expanded by -frewrite-includes */{{$}}
// CHECK-NEXT: # 12 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
// CHECK-NEXT: #pragma clang module import dummy /* clang -frewrite-includes: implicit import */{{$}}
// CHECK-NEXT: # 13 "{{.*[/\\]}}rewrite-includes-modules.c"{{$}}
