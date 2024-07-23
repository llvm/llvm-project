// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64 -emit-llvm  -target-cpu generic -target-feature +v8.5a %s -o - \
// RUN:                               | FileCheck %s --check-prefix=CHECK

__attribute__ ((target("branch-protection=none")))
void none() {}
// CHECK: define{{.*}} void @none() #[[#NONE:]]

  __attribute__ ((target("branch-protection=standard")))
void std() {}
// CHECK: define{{.*}} void @std() #[[#STD:]]

__attribute__ ((target("branch-protection=bti")))
void btionly() {}
// CHECK: define{{.*}} void @btionly() #[[#BTI:]]

__attribute__ ((target("branch-protection=pac-ret")))
void paconly() {}
// CHECK: define{{.*}} void @paconly() #[[#PAC:]]

__attribute__ ((target("branch-protection=pac-ret+bti")))
void pacbti0() {}
// CHECK: define{{.*}} void @pacbti0() #[[#PACBTI:]]

__attribute__ ((target("branch-protection=bti+pac-ret")))
void pacbti1() {}
// CHECK: define{{.*}} void @pacbti1() #[[#PACBTI]]

__attribute__ ((target("branch-protection=pac-ret+leaf")))
void leaf() {}
// CHECK: define{{.*}} void @leaf() #[[#PACLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+b-key")))
void bkey() {}
// CHECK: define{{.*}} void @bkey() #[[#PACBKEY:]]

__attribute__ ((target("branch-protection=pac-ret+b-key+leaf")))
void bkeyleaf0() {}
// CHECK: define{{.*}} void @bkeyleaf0()  #[[#PACBKEYLEAF:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key")))
void bkeyleaf1() {}
// CHECK: define{{.*}} void @bkeyleaf1()  #[[#PACBKEYLEAF]]

__attribute__ ((target("branch-protection=pac-ret+leaf+bti")))
void btileaf() {}
// CHECK: define{{.*}} void @btileaf() #[[#BTIPACLEAF:]]


__attribute__ ((target("branch-protection=pac-ret+pc")))
void pauthlr() {}
// CHECK: define{{.*}} void @pauthlr()  #[[#PAUTHLR:]]

__attribute__ ((target("branch-protection=pac-ret+pc+b-key")))
void pauthlr_bkey() {}
// CHECK: define{{.*}} void @pauthlr_bkey()  #[[#PAUTHLR_BKEY:]]

__attribute__ ((target("branch-protection=pac-ret+pc+leaf")))
void pauthlr_leaf() {}
// CHECK: define{{.*}} void @pauthlr_leaf()  #[[#PAUTHLR_LEAF:]]

__attribute__ ((target("branch-protection=pac-ret+pc+bti")))
void pauthlr_bti() {}
// CHECK: define{{.*}} void @pauthlr_bti()  #[[#PAUTHLR_BTI:]]

__attribute__ ((target("branch-protection=gcs")))
void gcs() {}
// CHECK: define{{.*}} void @gcs() #[[#GCS:]]

// CHECK-DAG: attributes #[[#NONE]] = { {{.*}}

// CHECK-DAG: attributes #[[#STD]] = { {{.*}} "branch-target-enforcement" "guarded-control-stack" {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#BTI]] = { {{.*}} "branch-target-enforcement"

// CHECK-DAG: attributes #[[#PAC]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACLEAF]] = { {{.*}} "sign-return-address"="all" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PACBKEY]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#PACBKEYLEAF]] = { {{.*}} "sign-return-address"="all" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#BTIPACLEAF]] = { {{.*}} "branch-target-enforcement" {{.*}}"sign-return-address"="all" "sign-return-address-key"="a_key"


// CHECK-DAG: attributes #[[#PAUTHLR]] = { {{.*}} "branch-protection-pauth-lr" {{.*}}"sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PAUTHLR_BKEY]] = { {{.*}} "branch-protection-pauth-lr" {{.*}}"sign-return-address"="non-leaf" "sign-return-address-key"="b_key"

// CHECK-DAG: attributes #[[#PAUTHLR_LEAF]] = { {{.*}} "branch-protection-pauth-lr" {{.*}}"sign-return-address"="all" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#PAUTHLR_BTI]] = { {{.*}} "branch-protection-pauth-lr" {{.*}}"branch-target-enforcement" {{.*}}"sign-return-address"="non-leaf" "sign-return-address-key"="a_key"

// CHECK-DAG: attributes #[[#GCS]] = { {{.*}} "guarded-control-stack"
