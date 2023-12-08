// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-apple-driverkit19.0 -S -o - %s | FileCheck %s

int main() { return 0; }
// CHECK: .build_version driverkit, 19, 0
