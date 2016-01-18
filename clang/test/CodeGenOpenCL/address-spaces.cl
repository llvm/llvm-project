// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck %s

void f__p(__private int *arg) { }
// CHECK: i32* nocapture %arg

void f__g(__global int *arg) { }
// CHECK: i32 addrspace(1)* nocapture %arg

void f__l(__local int *arg) { }
// CHECK: i32 addrspace(2)* nocapture %arg

void f__c(__constant int *arg) { }
// CHECK: i32 addrspace(3)* nocapture %arg


void fp(private int *arg) { }
// CHECK: i32* nocapture %arg

void fg(global int *arg) { }
// CHECK: i32 addrspace(1)* nocapture %arg

void fl(local int *arg) { }
// CHECK: i32 addrspace(2)* nocapture %arg

void fc(constant int *arg) { }
// CHECK: i32 addrspace(3)* nocapture %arg

