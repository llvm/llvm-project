// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// RUN: %clang -target i386-apple-darwin10 -mwatchsimulator-version-min=2.0 -arch i386 -S -o - %s | FileCheck %s
// RUN: %clang -target armv7s-apple-darwin10 -mwatchos-version-min=2.0 -arch armv7k -S -o - %s | FileCheck %s

int main() { return 0; }
// CHECK: .watchos_version_min 2, 0
