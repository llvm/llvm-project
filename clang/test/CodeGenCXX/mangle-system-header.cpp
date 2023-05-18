// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

// PR5420

# 1 "fake_system_header.h" 1 3 4
// CHECK-LABEL: define{{.*}} void @_ZdlPvS_(
void operator delete (void*, void*) {}

// PR6217
// CHECK-LABEL: define{{.*}} void @_Z3barv() 
void bar() { }
