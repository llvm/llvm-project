__attribute__((section("__RODATA,ThisSectionNameIsTooLong")))
int foo;

__attribute__((section("__RODATA,SectOK")))
int bar;

// RUN: %clang_cc1 -fhash-long-section-names=16 -emit-llvm -o - %s | FileCheck %s
// REQUIRES: system-darwin
// CHECK: @foo = global i32 0, section "__RODATA,ip9RNVxH27rCS+Ix"
// CHECK: @bar = global i32 0, section "__RODATA,SectOK"
