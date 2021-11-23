__attribute__((section("ThisSectionNameIsTooLong")))
int foo;

__attribute__((section("SectOK")))
int bar;

// RUN: %clang_cc1 -fhash-long-section-names=16 -emit-llvm -o - %s | FileCheck %s
// REQUIRES: x86_64-linux
// CHECK: @foo = global i32 0, section "ip9RNVxH27rCS+Ix"
// CHECK: @bar = global i32 0, section "SectOK"
