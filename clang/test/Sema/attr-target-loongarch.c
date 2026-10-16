// RUN: %clang_cc1 -triple loongarch64-linux-gnu  -fsyntax-only -verify %s

// expected-error@+1 {{function multiversioning is not supported on the current target}}
void __attribute__((target("default"))) bar(void) {}

// expected-error@+1 {{target(arch=..) attribute is not supported on targets missing invalid; specify an appropriate -march= or -mcpu=}}
void __attribute__((target("arch=invalid"))) foo(void) {}

// expected-warning@+1 {{unsupported '+div32' in the 'target' attribute string; 'target' attribute ignored}}
void __attribute__((target("+div32"))) plusfeature(void) {}

// expected-warning@+1 {{unsupported '-div32' in the 'target' attribute string; 'target' attribute ignored}}
void __attribute__((target("-div32"))) minusfeature(void) {}

// expected-warning@+1 {{unsupported 'aaa' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("aaa"))) test_feature(void) { return 4; }

// expected-warning@+1 {{unsupported 'aaa' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("no-aaa"))) test_nofeature(void) { return 4; }

// expected-warning@+1 {{duplicate 'arch=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=la464,arch=la664"))) test_duplarch(void) { return 4; }

// expected-warning@+1 {{unknown tune CPU 'la64v1.0' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=la64v1.0"))) test_tune(void) { return 4; }
