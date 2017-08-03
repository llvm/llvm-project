// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify %s

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo() { return 4; }
int __attribute__((target())) bar() { return 4; } //expected-error {{'target' attribute takes one argument}}
int __attribute__((target("tune=sandybridge"))) baz() { return 4; } //expected-warning {{Ignoring unsupported 'tune=' in the target attribute string}}
int __attribute__((target("fpmath=387"))) walrus() { return 4; } //expected-warning {{Ignoring unsupported 'fpmath=' in the target attribute string}}


