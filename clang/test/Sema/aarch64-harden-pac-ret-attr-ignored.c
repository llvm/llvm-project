// RUN: %clang_cc1 -triple aarch64 -target-cpu generic -target-feature +v8.5a -emit-llvm -verify %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-cpu generic -target-feature +v8.5a -msign-return-address=all -mharden-pac-ret=none -emit-llvm -verify %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-cpu generic -target-feature +v8.5a -msign-return-address=all -mharden-pac-ret=load-return-address -emit-llvm -verify %s -o - | FileCheck %s

// Invalid harden-pac-ret usage must cause the entire target attribute to be
// ignored, including its CPU and branch-protection settings. These functions
// must therefore share the baseline's attributes under each command-line mode.
void baseline(void) {}
// CHECK: define{{.*}} void @baseline() #[[#BASELINE:]] {

__attribute__((target("cpu=neoverse-v2,branch-protection=bti,harden-pac-ret=load-return-address")))
// expected-warning@-1 {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void bti_without_pac_ret(void) {}
// CHECK: define{{.*}} void @bti_without_pac_ret() #[[#BASELINE]] {

__attribute__((target("cpu=neoverse-v2,harden-pac-ret=load-return-address")))
// expected-warning@-1 {{'harden-pac-ret' attribute requires 'branch-protection=pac-ret'; 'target' attribute ignored}}
void missing_branch_protection(void) {}
// CHECK: define{{.*}} void @missing_branch_protection() #[[#BASELINE]] {
