// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -o - | FileCheck %s

// The following test that the function attributes take precedence over command-line options
// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -msign-return-address=all -mharden-pac-ret=none -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -emit-llvm -target-cpu generic -target-feature +v8.5a %s -msign-return-address=all -mharden-pac-ret=load-return-address -o - | FileCheck %s

__attribute__ ((target("branch-protection=none")))
void f1() {}
// CHECK: define{{.*}} void @f1() #[[#A1:]]

__attribute__ ((target("branch-protection=pac-ret")))
void f2() {}
// CHECK: define{{.*}} void @f2() #[[#A2:]]

__attribute__ ((target("branch-protection=pac-ret,harden-pac-ret=none")))
void f3() {}
// CHECK: define{{.*}} void @f3() #[[#A2]]

__attribute__ ((target("branch-protection=pac-ret,harden-pac-ret=load-return-address")))
void f4() {}
// CHECK: define{{.*}} void @f4() #[[#A3:]]

__attribute__ ((target("branch-protection=pac-ret+leaf")))
void f5() {}
// CHECK: define{{.*}} void @f5() #[[#A4:]]

__attribute__ ((target("branch-protection=pac-ret+leaf,harden-pac-ret=none")))
void f6() {}
// CHECK: define{{.*}} void @f6() #[[#A4]]

__attribute__ ((target("branch-protection=pac-ret+leaf,harden-pac-ret=load-return-address")))
void f7() {}
// CHECK: define{{.*}} void @f7() #[[#A5:]]

__attribute__ ((target("branch-protection=pac-ret+b-key")))
void f8() {}
// CHECK: define{{.*}} void @f8() #[[#A6:]]

__attribute__ ((target("branch-protection=pac-ret+b-key,harden-pac-ret=none")))
void f9() {}
// CHECK: define{{.*}} void @f9() #[[#A6]]

__attribute__ ((target("branch-protection=pac-ret+b-key,harden-pac-ret=load-return-address")))
void f10() {}
// CHECK: define{{.*}} void @f10() #[[#A7:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key")))
void f11() {}
// CHECK: define{{.*}} void @f11() #[[#A8:]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key,harden-pac-ret=none")))
void f12() {}
// CHECK: define{{.*}} void @f12() #[[#A8]]

__attribute__ ((target("branch-protection=pac-ret+leaf+b-key,harden-pac-ret=load-return-address")))
void f13() {}
// CHECK: define{{.*}} void @f13() #[[#A9:]]

// These check patterns rely on the fact that "sign-return-address-harden" appears after "sign-return-address"

// CHECK:     attributes #[[#A1]]
// CHECK-NOT: "sign-return-address"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     attributes #[[#A2]]
// CHECK:     "sign-return-address"="non-leaf"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     attributes #[[#A3]]
// CHECK:     "sign-return-address"="non-leaf"
// CHECK:     "sign-return-address-harden"="load-return-address"
// CHECK:     attributes #[[#A4]]
// CHECK:     "sign-return-address"="all"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     attributes #[[#A5]]
// CHECK:     "sign-return-address"="all"
// CHECK:     "sign-return-address-harden"="load-return-address"
// CHECK:     attributes #[[#A6]]
// CHECK:     "sign-return-address"="non-leaf"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#A7]]
// CHECK:     "sign-return-address"="non-leaf"
// CHECK:     "sign-return-address-harden"="load-return-address"
// CHECK:     "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#A8]]
// CHECK:     "sign-return-address"="all"
// CHECK-NOT: "sign-return-address-harden"
// CHECK:     "sign-return-address-key"="b_key"
// CHECK:     attributes #[[#A9]]
// CHECK:     "sign-return-address"="all"
// CHECK:     "sign-return-address-harden"="load-return-address"
// CHECK:     "sign-return-address-key"="b_key"
