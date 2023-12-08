// RUN: %clang_cc1 -triple mips-linux-gnu -emit-llvm  -o  - %s | FileCheck %s

void __attribute__((micromips)) foo (void) {}

// CHECK: define{{.*}} void @foo() [[MICROMIPS:#[0-9]+]]

void __attribute__((nomicromips)) nofoo (void) {}

// CHECK: define{{.*}} void @nofoo() [[NOMICROMIPS:#[0-9]+]]

// CHECK: attributes [[MICROMIPS]] = { noinline nounwind {{.*}} "micromips" {{.*}} }
// CHECK: attributes [[NOMICROMIPS]]  = { noinline nounwind {{.*}} "nomicromips" {{.*}} }
