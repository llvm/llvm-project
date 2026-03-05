// RUN: %clang_cc1 -triple aarch64 -emit-llvm  -target-cpu generic -target-feature +v8.5a %s -o - \
// RUN: | FileCheck %s --check-prefix=CHECK

__attribute__ ((target("branch-protection=pac-ret,harden-pac-ret=none")))
void f1() {}
// CHECK: define{{.*}} void @f1() #[[#F1:]]

__attribute__ ((target("branch-protection=pac-ret,harden-pac-ret=load-return-address")))
void f2() {}
// CHECK: define{{.*}} void @f2() #[[#F2:]]

__attribute__ ((target("branch-protection=pac-ret+leaf,harden-pac-ret=none")))
void f3() {}
// CHECK: define{{.*}} void @f3() #[[#F3:]]

__attribute__ ((target("branch-protection=pac-ret+leaf,harden-pac-ret=load-return-address")))
void f4() {}
// CHECK: define{{.*}} void @f4() #[[#F4:]]

__attribute__ ((target("branch-protection=pac-ret+b-key,harden-pac-ret=none")))
void f5() {}
// CHECK: define{{.*}} void @f5() #[[#F5:]]

__attribute__ ((target("branch-protection=pac-ret+b-key,harden-pac-ret=load-return-address")))
void f6() {}
// CHECK: define{{.*}} void @f6() #[[#F6:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key,harden-pac-ret=none")))
void f7() {}
// CHECK: define{{.*}} void @f7() #[[#F7:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key,harden-pac-ret=load-return-address")))
void f8() {}
// CHECK: define{{.*}} void @f8() #[[#F8:]]

// CHECK-DAG: attributes #[[#F1]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="none"
// CHECK-DAG: attributes #[[#F2]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="load-return-address"
// CHECK-DAG: attributes #[[#F3]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="none"
// CHECK-DAG: attributes #[[#F4]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="load-return-address"
// CHECK-DAG: attributes #[[#F5]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="none" "sign-return-address-key"="b_key"
// CHECK-DAG: attributes #[[#F6]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="load-return-address" "sign-return-address-key"="b_key"
// CHECK-DAG: attributes #[[#F7]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="none" "sign-return-address-key"="b_key"
// CHECK-DAG: attributes #[[#F8]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="load-return-address" "sign-return-address-key"="b_key"
