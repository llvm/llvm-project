// RUN: %clang_cc1 -triple thumbv6m -verify -fsyntax-only %s

// expected-no-diagnostics
// Armv8.1-M.Main
__attribute__((target("arch=cortex-m55,branch-protection=bti"))) void f1() {}
__attribute__((target("arch=cortex-m55,branch-protection=pac-ret"))) void f2() {}
__attribute__((target("arch=cortex-m55,branch-protection=bti+pac-ret"))) void f3() {}
__attribute__((target("arch=cortex-m55,branch-protection=bti+pac-ret+leaf"))) void f4() {}
// Armv8-M.Main
__attribute__((target("arch=cortex-m33,branch-protection=bti"))) void f5() {}
__attribute__((target("arch=cortex-m33,branch-protection=pac-ret"))) void f6() {}
__attribute__((target("arch=cortex-m33,branch-protection=bti+pac-ret"))) void f7() {}
__attribute__((target("arch=cortex-m33,branch-protection=bti+pac-ret+leaf"))) void f8() {}
// Armv7-M
__attribute__((target("arch=cortex-m3,branch-protection=bti"))) void f9() {}
__attribute__((target("arch=cortex-m3,branch-protection=pac-ret"))) void f10() {}
__attribute__((target("arch=cortex-m3,branch-protection=bti+pac-ret"))) void f11() {}
__attribute__((target("arch=cortex-m3,branch-protection=bti+pac-ret+leaf"))) void f12() {}
// Armv7E-M
__attribute__((target("arch=cortex-m4,branch-protection=bti"))) void f13() {}
__attribute__((target("arch=cortex-m4,branch-protection=pac-ret"))) void f14() {}
__attribute__((target("arch=cortex-m4,branch-protection=bti+pac-ret"))) void f15() {}
__attribute__((target("arch=cortex-m4,branch-protection=bti+pac-ret+leaf"))) void f16() {}
