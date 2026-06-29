// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -o - | FileCheck %s

// The following test that the function attributes take precedence over command-line options
// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -msign-return-address=all -mharden-pac-ret=none -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -msign-return-address=all -mharden-pac-ret=load-return-address -o - | FileCheck %s

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

// These check patterns rely on the fact that "sign-return-address-harden" appears after "sign-return-address"

// CHECK:     attributes #[[#F1]] = { {{.*}} "sign-return-address"="non-leaf"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     attributes #[[#F2]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="load-return-address"
// CHECK:     attributes #[[#F3]] = { {{.*}} "sign-return-address"="all"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     attributes #[[#F4]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="load-return-address"
// CHECK:     attributes #[[#F5]] = { {{.*}} "sign-return-address"="non-leaf"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#F6]] = { {{.*}} "sign-return-address"="non-leaf" "sign-return-address-harden"="load-return-address" "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#F7]] = { {{.*}} "sign-return-address"="all"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#F8]] = { {{.*}} "sign-return-address"="all" "sign-return-address-harden"="load-return-address" "sign-return-address-key"="b_key"
