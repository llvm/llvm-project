// The 8-bit integer matrix multiply extension is a mandatory component of the
// Armv8.6-A extensions, but is permitted as an optional feature for any
// implementation of Armv8.2-A to Armv8.5-A (inclusive)
// RUN: %clang -target aarch64 -march=armv8.5a -S -o - -emit-llvm -c %s 2>&1 | FileCheck -check-prefix=NOI8MM %s
// RUN: %clang -target aarch64 -march=armv8.5a+i8mm -S -o - -emit-llvm -c %s 2>&1 | FileCheck -check-prefix=I8MM %s
// RUN: %clang -target aarch64 -march=armv8.6a -S -o - -emit-llvm -c %s 2>&1 | FileCheck -check-prefix=I8MM %s

// I8MM: "target-features"="{{.*}},+i8mm
// NOI8MM-NOT: "target-features"="{{.*}},+i8mm

void test() {}
