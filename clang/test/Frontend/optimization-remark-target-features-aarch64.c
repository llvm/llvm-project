// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64-unknown-linux-gnu --rtlib=compiler-rt \
// RUN:   -S -O0 -Rpass-analysis=target-features -fno-caret-diagnostics %s -o /dev/null 2>&1 | FileCheck %s

int baseline(void) { return 0; }
__attribute__((target("sve2"))) int targeted(void) { return 1; }
__attribute__((target_clones("sve2", "default"))) int clones(void) { return 2; }
__attribute__((target_version("sve2"))) int versioned(void) { return 3; }
__attribute__((target_version("default"))) int versioned(void) { return 4; }
int caller(void) { return clones() + versioned(); }

// Inspect the subtarget used for each emitted function, including FMV variants
// and their resolvers. Default functions must not inherit variant features.
// CHECK-NOT: Enabled features
// CHECK: {{.*}}remark: Enabled features for @baseline:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @targeted: {{([^,]+,)*}}fp-armv8,fullfp16,{{([^,]+,)*}}sve,sve2{{(,.*)?}} [-Rpass-analysis=target-features]
// CHECK-NEXT: {{.*}}remark: Enabled features for @clones._Msve2: {{([^,]+,)*}}fp-armv8,fullfp16,{{([^,]+,)*}}sve,sve2{{(,.*)?}} [-Rpass-analysis=target-features]
// CHECK-NEXT: {{.*}}remark: Enabled features for @clones.default:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @versioned._Msve2: {{([^,]+,)*}}fp-armv8,fullfp16,{{([^,]+,)*}}sve,sve2{{(,.*)?}} [-Rpass-analysis=target-features]
// CHECK-NEXT: {{.*}}remark: Enabled features for @versioned.default:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @caller:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @clones.resolver:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @versioned.resolver:
// CHECK-NOT: {{(^|[ ,])(sve|sve2)(,| |$)}}
// CHECK-NOT: Enabled features
