// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fno-autolink -o - -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s | FileCheck --check-prefix=CHECK-AUTOLINK-DISABLED %s

@import AutolinkTBD;

int f() {
  return foo();
}

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 6, !"Linker Options", ![[AUTOLINK_OPTIONS:[0-9]+]]}
// CHECK: ![[AUTOLINK_OPTIONS]] = !{![[AUTOLINK_FRAMEWORK:[0-9]+]]}
// CHECK: ![[AUTOLINK_FRAMEWORK]] = !{!"-framework", !"AutolinkTBD"}

// CHECK-AUTOLINK-DISABLED: !llvm.module.flags
// CHECK-AUTOLINK-DISABLED-NOT: "Linker Options"
