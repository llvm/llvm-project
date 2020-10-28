// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-unknown-unknown-eabi -emit-llvm  -target-cpu generic -target-feature +v8.5a %s -o - \
// RUN:                               | FileCheck %s --check-prefix=CHECK

__attribute__ ((target("branch-protection=none")))
void none() {}
// CHECK: define void @none() #[[#NONE:]]

  __attribute__ ((target("branch-protection=standard")))
void std() {}
// CHECK: define void @std() #[[#STD:]]

__attribute__ ((target("branch-protection=bti")))
void btionly() {}
// CHECK: define void @btionly() #[[#BTI:]]

__attribute__ ((target("branch-protection=pac-ret")))
void paconly() {}
// CHECK: define void @paconly() #[[#PAC:]]

__attribute__ ((target("branch-protection=pac-ret+bti")))
void pacbti0() {}
// CHECK: define void @pacbti0() #[[#PACBTI:]]

__attribute__ ((target("branch-protection=bti+pac-ret")))
void pacbti1() {}
// CHECK: define void @pacbti1() #[[#PACBTI]]

__attribute__ ((target("branch-protection=pac-ret+leaf")))
void leaf() {}
// CHECK: define void @leaf() #[[#PACLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+b-key")))
void bkey() {}
// CHECK: define void @bkey() #[[#PACBKEY:]]

__attribute__ ((target("branch-protection=pac-ret+b-key+leaf")))
void bkeyleaf0() {}
// CHECK: define void @bkeyleaf0()  #[[#PACBKEYLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key")))
void bkeyleaf1() {}
// CHECK: define void @bkeyleaf1()  #[[#PACBKEYLEAF]]

__attribute__ ((target("branch-protection=pac-ret+leaf+bti")))
void btileaf() {}
// CHECK: define void @btileaf() #[[#BTIPACLEAF:]]

// CHECK-DAG: attributes #[[#NONE]] = { {{.*}} "branch-target-enforcement"="false" {{.*}} "sign-return-address"="none"

// CHECK-DAG: attributes #[[#STD]] = { {{.*}} "branch-target-enforcement"="true" {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#BTI]] = { {{.*}} "branch-target-enforcement"="true" {{.*}} "sign-return-address"="none"

// CHECK-DAG: attributes #[[#PAC]] = { {{.*}} "branch-target-enforcement"="false" {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACLEAF]] = { {{.*}} "branch-target-enforcement"="false" {{.*}}"sign-return-address"="all" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACBKEY]] = { {{.*}}"branch-target-enforcement"="false" {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#PACBKEYLEAF]] = { {{.*}} "branch-target-enforcement"="false" {{.*}}"sign-return-address"="all" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#BTIPACLEAF]] = { {{.*}}"branch-target-enforcement"="true" {{.*}} "sign-return-address"="all" "sign-return-address-key"="a_key"
