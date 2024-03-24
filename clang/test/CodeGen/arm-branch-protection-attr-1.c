// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7m-unknown-unknown-eabi -emit-llvm %s -o - \
// RUN:                               | FileCheck %s --check-prefix=CHECK

__attribute__((target("branch-protection=none"))) void none() {}
// CHECK: define{{.*}} void @none() #[[#NONE:]]

__attribute__((target("branch-protection=standard"))) void std() {}
// CHECK: define{{.*}} void @std() #[[#STD:]]

__attribute__((target("branch-protection=bti"))) void btionly() {}
// CHECK: define{{.*}} void @btionly() #[[#BTI:]]

__attribute__((target("branch-protection=pac-ret"))) void paconly0() {}
// CHECK: define{{.*}} void @paconly0() #[[#PAC:]]

__attribute__((target("branch-protection=pac-ret+b-key"))) void paconly1() {}
// CHECK: define{{.*}} void @paconly1() #[[#PAC]]

__attribute__((target("branch-protection=pac-ret+bti"))) void pacbti0() {}
// CHECK: define{{.*}} void @pacbti0() #[[#STD]]

__attribute__((target("branch-protection=bti+pac-ret"))) void pacbti1() {}
// CHECK: define{{.*}} void @pacbti1() #[[#STD]]

__attribute__((target("branch-protection=pac-ret+leaf"))) void leaf() {}
// CHECK: define{{.*}} void @leaf() #[[#PACLEAF:]]

__attribute__((target("branch-protection=pac-ret+leaf+bti"))) void btileaf() {}
// CHECK: define{{.*}} void @btileaf() #[[#BTIPACLEAF:]]

// CHECK-DAG: attributes #[[#NONE]] = { {{.*}}

// CHECK-DAG: attributes #[[#STD]] = { {{.*}} "branch-target-enforcement" {{.*}} "sign-return-address"="non-leaf"

// CHECK-DAG: attributes #[[#BTI]] = { {{.*}} "branch-target-enforcement"

// CHECK-DAG: attributes #[[#PAC]] = { {{.*}} "sign-return-address"="non-leaf"

// CHECK-DAG: attributes #[[#PACLEAF]] = { {{.*}} "sign-return-address"="all"

// CHECK-DAG: attributes #[[#BTIPACLEAF]] = { {{.*}} "branch-target-enforcement" {{.*}} "sign-return-address"="all"
