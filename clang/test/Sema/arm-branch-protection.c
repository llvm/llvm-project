// RUN: %clang_cc1 -triple thumbv6m -verify -fsyntax-only %s

// expected-no-diagnostics
// Armv8.1-M.Main
__attribute__((target("arch=cortex-m55,branch-protection=bti"))) void f1(void) {}
__attribute__((target("arch=cortex-m55,branch-protection=pac-ret"))) void f2(void) {}
__attribute__((target("arch=cortex-m55,branch-protection=bti+pac-ret"))) void f3(void) {}
__attribute__((target("arch=cortex-m55,branch-protection=bti+pac-ret+leaf"))) void f4(void) {}
// Armv8-M.Main
__attribute__((target("arch=cortex-m33,branch-protection=bti"))) void f5(void) {}
__attribute__((target("arch=cortex-m33,branch-protection=pac-ret"))) void f6(void) {}
__attribute__((target("arch=cortex-m33,branch-protection=bti+pac-ret"))) void f7(void) {}
__attribute__((target("arch=cortex-m33,branch-protection=bti+pac-ret+leaf"))) void f8(void) {}
// Armv7-M
__attribute__((target("arch=cortex-m3,branch-protection=bti"))) void f9(void) {}
__attribute__((target("arch=cortex-m3,branch-protection=pac-ret"))) void f10(void) {}
__attribute__((target("arch=cortex-m3,branch-protection=bti+pac-ret"))) void f11(void) {}
__attribute__((target("arch=cortex-m3,branch-protection=bti+pac-ret+leaf"))) void f12(void) {}
// Armv7E-M
__attribute__((target("arch=cortex-m4,branch-protection=bti"))) void f13(void) {}
__attribute__((target("arch=cortex-m4,branch-protection=pac-ret"))) void f14(void) {}
__attribute__((target("arch=cortex-m4,branch-protection=bti+pac-ret"))) void f15(void) {}
__attribute__((target("arch=cortex-m4,branch-protection=bti+pac-ret+leaf"))) void f16(void) {}
