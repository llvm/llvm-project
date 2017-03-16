// RUN: %clang -save-temps -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header -emit-llvm -S -### %s
// CHECK-NOT: finclude-default-header
// Make sure we don't pass -finclude-default-header to any commands other than the driver.

void test() {}

