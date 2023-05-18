// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// RUN: %clang -target i386-apple-darwin10 -miphonesimulator-version-min=7.0 -arch i386 -S -o - %s | FileCheck %s
// RUN: %clang -target armv7s-apple-darwin10 -miphoneos-version-min=7.0 -arch armv7s -S -o - %s | FileCheck %s

int main() { return 0; }
// CHECK: .ios_version_min 7, 0
