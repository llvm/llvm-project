// RUN: %clang_cc1 -triple thumbv6m -verify -fsyntax-only %s

// expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
__attribute__((target("arch=cortex-m0,branch-protection=bti"))) void f1() {}

// expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
__attribute__((target("arch=cortex-m0,branch-protection=pac-ret"))) void f2() {}

// expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
__attribute__((target("arch=cortex-m0,branch-protection=bti+pac-ret"))) void f3() {}

// expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
__attribute__((target("arch=cortex-m0,branch-protection=bti+pac-ret+leaf"))) void f4() {}

// expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
__attribute__((target("arch=cortex-a17,thumb,branch-protection=bti+pac-ret+leaf"))) void f5() {}
