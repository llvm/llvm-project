// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify -std=c2x %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify -std=c2x %s
// RUN: %clang_cc1 -triple arm-linux-gnu  -fsyntax-only -verify -std=c2x %s

#ifdef __x86_64__

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo(void) { return 4; }
//expected-error@+1 {{'target' attribute takes one argument}}
int __attribute__((target())) bar(void) { return 4; }
// no warning, tune is supported for x86
int __attribute__((target("tune=sandybridge"))) baz(void) { return 4; }
//expected-warning@+1 {{unsupported 'fpmath=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("fpmath=387"))) walrus(void) { return 4; }
//expected-warning@+1 {{unknown CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("avx,sse4.2,arch=hiss"))) meow(void) {  return 4; }
//expected-warning@+1 {{unsupported 'woof' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("woof"))) bark(void) {  return 4; }
// no warning, same as saying 'nothing'.
int __attribute__((target("arch="))) turtle(void) { return 4; }
//expected-warning@+1 {{unknown CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=hiss,arch=woof"))) pine_tree(void) { return 4; }
//expected-warning@+1 {{duplicate 'arch=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=ivybridge,arch=haswell"))) oak_tree(void) { return 4; }
//expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("branch-protection=none"))) birch_tree(void) { return 5; }
//expected-warning@+1 {{unknown tune CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=hiss,tune=woof"))) apple_tree(void) { return 4; }

#elifdef __aarch64__

int __attribute__((target("sve,arch=armv8-a"))) foo(void) { return 4; }
//expected-error@+1 {{'target' attribute takes one argument}}
int __attribute__((target())) bar(void) { return 4; }
// no warning, tune is supported for aarch64
int __attribute__((target("tune=cortex-a710"))) baz(void) { return 4; }
//expected-warning@+1 {{unsupported 'fpmath=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("fpmath=387"))) walrus(void) { return 4; }
//expected-warning@+1 {{unknown CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("sve,cpu=hiss"))) meow(void) {  return 4; }
// FIXME: We currently have no implementation of isValidFeatureName, so this is not noticed as an error.
int __attribute__((target("woof"))) bark(void) {  return 4; }
// FIXME: Same
int __attribute__((target("arch=armv8-a+woof"))) buff(void) {  return 4; }
// FIXME: Same
int __attribute__((target("+noway"))) noway(void) {  return 4; }
// no warning, same as saying 'nothing'.
int __attribute__((target("arch="))) turtle(void) { return 4; }
//expected-warning@+1 {{unknown CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("cpu=hiss,cpu=woof"))) pine_tree(void) { return 4; }
//expected-warning@+1 {{duplicate 'arch=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=armv8.1-a,arch=armv8-a"))) oak_tree(void) { return 4; }
//expected-warning@+1 {{duplicate 'cpu=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("cpu=cortex-a710,cpu=neoverse-n2"))) apple_tree(void) { return 4; }
//expected-warning@+1 {{duplicate 'tune=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=cortex-a710,tune=neoverse-n2"))) pear_tree(void) { return 4; }
// no warning - branch-protection should work on aarch64
int __attribute__((target("branch-protection=none"))) birch_tree(void) { return 5; }

#else

// tune is not supported by other targets.
//expected-warning@+1 {{unsupported 'tune=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=hiss"))) baz(void) { return 4; }

#endif
