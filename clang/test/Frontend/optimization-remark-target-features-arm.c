// REQUIRES: arm-registered-target
// RUN: %clang --target=armv7-unknown-linux-gnueabi \
// RUN:   -S -O0 -Rpass-analysis=target-features -fno-caret-diagnostics %s -o /dev/null 2>&1 | FileCheck %s

int baseline(void) { return 0; }
__attribute__((target("thumb"))) int thumb_function(void) { return 1; }
__attribute__((target("arm"))) int arm_function(void) { return 2; }
int baseline_again(void) { return 3; }

// ARM and Thumb functions in the same translation unit use distinct subtargets.
// CHECK-NOT: Enabled features
// CHECK: {{.*}}remark: Enabled features for @baseline:
// CHECK-NOT: {{(^|[ ,])thumb-mode(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @thumb_function: {{([^,]+,)*}}thumb-mode{{(,.*)?}} [-Rpass-analysis=target-features]
// CHECK-NEXT: {{.*}}remark: Enabled features for @arm_function:
// CHECK-NOT: {{(^|[ ,])thumb-mode(,| |$)}}
// CHECK-NEXT: {{.*}}remark: Enabled features for @baseline_again:
// CHECK-NOT: {{(^|[ ,])thumb-mode(,| |$)}}
// CHECK-NOT: Enabled features
