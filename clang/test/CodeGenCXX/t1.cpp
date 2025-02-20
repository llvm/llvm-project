// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -fms-extensions -verify -o - %s | FileCheck %s

// expected-no-diagnostics

extern "C" {

#pragma section("read_flag_section", execute)
__declspec(allocate("read_flag_section")) int unreferenced = 0;
extern __declspec(allocate("read_flag_section")) int referenced = 42;
int *user() { return &referenced; }

}

//CHECK: @unreferenced = dso_local constant i32 0, section "read_flag_section"
//CHECK: @referenced = dso_local constant i32 42, section "read_flag_section"

