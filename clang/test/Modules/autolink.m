// UNSUPPORTED: -zos, -aix
// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-pch -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -o %t.pch -I %S/Inputs -x objective-c-header %S/Inputs/autolink-sub3.pch
// RUN: %clang_cc1 -emit-llvm -o - -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs -include-pch %t.pch %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fno-autolink -o - -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs -include-pch %t.pch %s | FileCheck --check-prefix=CHECK-AUTOLINK-DISABLED %s

@import autolink.sub2;

int f(void) {
  return autolink_sub2();
}

@import autolink;

int g(void) {
  return autolink;
}

@import Module.SubFramework;
const char *get_module_subframework(void) {
  return module_subframework;
}

@import DependsOnModule.SubFramework;
float *get_module_subframework_dep(void) {
  return sub_framework;
}

@import NoUmbrella;
int use_no_umbrella(void) {
  return no_umbrella_A;
}

int use_autolink_sub3(void) {
  return autolink_sub3();
}

// NOTE: "autolink_sub" is intentionally not linked.

// CHECK: !llvm.linker.options = !{![[AUTOLINK_PCH:[0-9]+]], ![[AUTOLINK_FRAMEWORK:[0-9]+]], ![[AUTOLINK:[0-9]+]], ![[DEPENDSONMODULE:[0-9]+]], ![[MODULE:[0-9]+]], ![[NOUMBRELLA:[0-9]+]]}
// CHECK: ![[AUTOLINK_PCH]] = !{!"{{(-l|/DEFAULTLIB:|lib", !")}}autolink_from_pch{{(\.lib)?}}"}
// CHECK: ![[AUTOLINK_FRAMEWORK]] = !{!"-framework", !"autolink_framework"}
// CHECK: ![[AUTOLINK]] = !{!"{{(-l|/DEFAULTLIB:|lib", !")}}autolink{{(\.lib)?}}"}
// CHECK: ![[DEPENDSONMODULE]] = !{!"-framework", !"DependsOnModule"}
// CHECK: ![[MODULE]] = !{!"-framework", !"Module"}
// CHECK: ![[NOUMBRELLA]] = !{!"-framework", !"NoUmbrella"}

// CHECK-AUTOLINK-DISABLED: !llvm.module.flags
// CHECK-AUTOLINK-DISABLED-NOT: !llvm.linker.options
