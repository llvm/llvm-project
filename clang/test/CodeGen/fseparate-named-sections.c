// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-linux -S -o - < %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux -S -fseparate-named-sections -o - < %s | FileCheck %s --check-prefix=SEPARATE

__attribute__((section("custom_text"))) void f(void) {}
__attribute__((section("custom_text"))) void g(void) {}

// CHECK: .section custom_text,"ax",@progbits{{$}}
// CHECK: f:
// CHECK: g:

// SEPARATE: .section custom_text,"ax",@progbits,unique,1{{$}}
// SEPARATE: f:
// SEPARATE: .section custom_text,"ax",@progbits,unique,2{{$}}
// SEPARATE: g:

__attribute__((section("custom_data"))) int i = 0;
__attribute__((section("custom_data"))) int j = 0;

// CHECK: .section custom_data,"aw",@progbits{{$}}
// CHECK: i:
// CHECK: j:

// SEPARATE: .section custom_data,"aw",@progbits,unique,3{{$}}
// SEPARATE: i:
// SEPARATE: .section custom_data,"aw",@progbits,unique,4{{$}}
// SEPARATE: j:
