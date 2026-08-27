// Verify that implicit conversions of DriverKit26 map to DriverKit27.

// RUN: rm -rf %t
// RUN: %clang -target arm64-apple-driverkit26 %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: overriding deployment version from '26' to '27'
// CHECK: 'f0' is deprecated: first deprecated in DriverKit 27.0

void f0(int) __attribute__((availability(driverkit,introduced=19.0,deprecated=26.0)));
void test() {
  f0(0);
}
