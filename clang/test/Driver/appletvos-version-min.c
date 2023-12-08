// REQUIRES: x86-registered-target
// REQUIRES: aarch64-registered-target
// RUN: %clang -target i386-apple-darwin10 -mappletvsimulator-version-min=9.0 -arch x86_64 -S -o - %s | FileCheck %s
// RUN: %clang -target armv7s-apple-darwin10 -mappletvos-version-min=9.0 -arch arm64 -S -o - %s | FileCheck %s
// RUN: env TVOS_DEPLOYMENT_TARGET=9.0 %clang -isysroot SDKs/MacOSX10.9.sdk -target i386-apple-darwin10  -arch x86_64 -S -o - %s | FileCheck %s

int main() { return 0; }
// CHECK: .tvos_version_min 9, 0
