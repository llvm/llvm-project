// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify %s

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo() { return 4; }
int __attribute__((target())) bar() { return 4; } //expected-error {{'target' attribute takes one argument}}
int __attribute__((target("tune=sandybridge"))) baz() { return 4; } //expected-warning {{ignoring unsupported 'tune=' in the target attribute string}}
int __attribute__((target("fpmath=387"))) walrus() { return 4; } //expected-warning {{ignoring unsupported 'fpmath=' in the target attribute string}}
int __attribute__((target("avx,sse4.2,arch=hiss"))) meow() {  return 4; }//expected-warning {{ignoring unsupported architecture 'hiss' in the target attribute string}}
int __attribute__((target("woof"))) bark() {  return 4; }//expected-warning {{ignoring unsupported 'woof' in the target attribute string}}
int __attribute__((target("arch="))) turtle() { return 4; } // no warning, same as saying 'nothing'.
int __attribute__((target("arch=hiss,arch=woof"))) pine_tree() { return 4; } //expected-warning {{ignoring unsupported architecture 'hiss' in the target attribute string}}
int __attribute__((target("arch=ivybridge,arch=haswell"))) oak_tree() { return 4; } //expected-warning {{ignoring duplicate 'arch=' in the target attribute string}}



