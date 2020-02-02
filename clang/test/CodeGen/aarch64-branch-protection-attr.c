// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-unknown-unknown-eabi -emit-llvm  -target-cpu generic -target-feature +v8.5a %s -o - \
// RUN:                               | FileCheck %s --check-prefix=CHECK --check-prefix=NO-OVERRIDE
// RUN: %clang_cc1 -triple aarch64-unknown-unknown-eabi -emit-llvm  -target-cpu generic -target-feature +v8.5a %s -o - \
// RUN:   -msign-return-address=non-leaf -msign-return-address-key=a_key -mbranch-target-enforce \
// RUN:                               | FileCheck %s --check-prefix=CHECK --check-prefix=OVERRIDE

void missing() {}
// NO-OVERRIDE: define dso_local void @missing() #[[#NONE:]]
// OVERRIDE: define dso_local void @missing() #[[#STD:]]

__attribute__ ((target("branch-protection=none")))
void none() {}
// NO-OVERRIDE: define dso_local void @none() #[[#NONE]]
// OVERRIDE: define dso_local void @none() #[[#NONE:]]

  __attribute__ ((target("branch-protection=standard")))
void std() {}
// NO-OVERRIDE: define dso_local void @std() #[[#STD:]]
// OVERRIDE: define dso_local void @std() #[[#STD]]

__attribute__ ((target("branch-protection=bti")))
void btionly() {}
// NO-OVERRIDE: define dso_local void @btionly() #[[#BTI:]]
// OVERRIDE: define dso_local void @btionly() #[[#BTI:]]

__attribute__ ((target("branch-protection=pac-ret")))
void paconly() {}
// NO-OVERRIDE: define dso_local void @paconly() #[[#PAC:]]
// OVERRIDE: define dso_local void @paconly() #[[#PAC:]]

__attribute__ ((target("branch-protection=pac-ret+bti")))
void pacbti0() {}
// NO-OVERRIDE: define dso_local void @pacbti0() #[[#PACBTI:]]
// OVERRIDE: define dso_local void @pacbti0() #[[#PACBTI:]]

__attribute__ ((target("branch-protection=bti+pac-ret")))
void pacbti1() {}
// NO-OVERRIDE: define dso_local void @pacbti1() #[[#PACBTI]]
// OVERRIDE: define dso_local void @pacbti1() #[[#PACBTI]]

__attribute__ ((target("branch-protection=pac-ret+leaf")))
void leaf() {}
// NO-OVERRIDE: define dso_local void @leaf() #[[#PACLEAF:]]
// OVERRIDE: define dso_local void @leaf() #[[#PACLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+b-key")))
void bkey() {}
// NO-OVERRIDE: define dso_local void @bkey() #[[#PACBKEY:]]
// OVERRIDE: define dso_local void @bkey() #[[#PACBKEY:]]

__attribute__ ((target("branch-protection=pac-ret+b-key+leaf")))
void bkeyleaf0() {}
// NO-OVERRIDE: define dso_local void @bkeyleaf0()  #[[#PACBKEYLEAF:]]
// OVERRIDE: define dso_local void @bkeyleaf0()  #[[#PACBKEYLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key")))
void bkeyleaf1() {}
// NO-OVERRIDE: define dso_local void @bkeyleaf1()  #[[#PACBKEYLEAF]]
// OVERRIDE: define dso_local void @bkeyleaf1()  #[[#PACBKEYLEAF]]

__attribute__ ((target("branch-protection=pac-ret+leaf+bti")))
void btileaf() {}
// NO-OVERRIDE: define dso_local void @btileaf() #[[#BTIPACLEAF:]]
// OVERRIDE: define dso_local void @btileaf() #[[#BTIPACLEAF:]]

// CHECK-DAG: attributes #[[#NONE]]

// CHECK-DAG: attributes #[[#STD]] = { {{.*}} "branch-target-enforcement" {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#BTI]] = { {{.*}}"branch-target-enforcement"

// CHECK-DAG: attributes #[[#PAC]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACLEAF]] = { {{.*}} "sign-return-address"="all" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACBKEY]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#PACBKEYLEAF]] = { {{.*}} "sign-return-address"="all" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#BTIPACLEAF]] = { {{.*}}"branch-target-enforcement" {{.*}} "sign-return-address"="all" "sign-return-address-key"="a_key"
