// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify %s

#ifdef __x86_64__

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo() { return 4; }
//expected-error@+1 {{'target' attribute takes one argument}}
int __attribute__((target())) bar() { return 4; }
// no warning, tune is supported for x86
int __attribute__((target("tune=sandybridge"))) baz() { return 4; }
//expected-warning@+1 {{unsupported 'fpmath=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("fpmath=387"))) walrus() { return 4; }
//expected-warning@+1 {{unknown architecture 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("avx,sse4.2,arch=hiss"))) meow() {  return 4; }
//expected-warning@+1 {{unsupported 'woof' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("woof"))) bark() {  return 4; }
// no warning, same as saying 'nothing'.
int __attribute__((target("arch="))) turtle() { return 4; }
//expected-warning@+1 {{unknown architecture 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=hiss,arch=woof"))) pine_tree() { return 4; }
//expected-warning@+1 {{duplicate 'arch=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=ivybridge,arch=haswell"))) oak_tree() { return 4; }
//expected-warning@+1 {{unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("branch-protection=none"))) birch_tree() { return 5; }
//expected-warning@+1 {{unknown tune CPU 'hiss' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=hiss,tune=woof"))) apple_tree() { return 4; }

#else

// tune is not supported by other targets.
//expected-warning@+1 {{unsupported 'tune=' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("tune=hiss"))) baz() { return 4; }

#endif
