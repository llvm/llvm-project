// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi -o - %s | FileCheck %s

// Test interaction between __attribute__((section())) and '#pragma clang
// section' directives. The attribute should always have higher priority than
// the pragma.

// Text tests.
#pragma clang section text=".ext_fun_pragma"
void ext_fun(void) __attribute__((section(".ext_fun_attr")));
void ext_fun(void) {}
#pragma clang section text=""

void ext_fun2(void) __attribute__((section(".ext_fun2_attr")));
#pragma clang section text=".ext_fun2_pragma"
void ext_fun2(void) {}
#pragma clang section text=""

#pragma clang section text=".int_fun_pragma"
static void int_fun(void) __attribute__((section(".int_fun_attr"), used));
static void int_fun(void) {}
#pragma clang section text=""

static void int_fun2(void) __attribute__((section(".int_fun2_attr"), used));
#pragma clang section text=".int_fun2_pragma"
static void int_fun2(void) {}
#pragma clang section text=""

// Rodata tests.
#pragma clang section rodata=".ext_const_pragma"
__attribute__((section(".ext_const_attr")))
const int ext_const = 1;
#pragma clang section rodata=""

#pragma clang section rodata=".int_const_pragma"
__attribute__((section(".int_const_attr"), used))
static const int int_const = 1;
#pragma clang section rodata=""

// Data tests.
#pragma clang section data=".ext_var_pragma"
__attribute__((section(".ext_var_attr")))
int ext_var = 1;
#pragma clang section data=""

#pragma clang section data=".int_var_pragma"
__attribute__((section(".int_var_attr"), used))
static int int_var = 1;
#pragma clang section data=""

// Bss tests.
#pragma clang section bss=".ext_zvar_pragma"
__attribute__((section(".ext_zvar_attr")))
int ext_zvar;
#pragma clang section bss=""

#pragma clang section bss=".int_zvar_pragma"
__attribute__((section(".int_zvar_attr"), used))
static int int_zvar;
#pragma clang section bss=""

// CHECK: @ext_const = constant i32 1, section ".ext_const_attr", align 4{{$}}
// CHECK: @int_const = internal constant i32 1, section ".int_const_attr", align 4{{$}}
// CHECK: @ext_var = global i32 1, section ".ext_var_attr", align 4{{$}}
// CHECK: @int_var = internal global i32 1, section ".int_var_attr", align 4{{$}}
// CHECK: @ext_zvar = global i32 0, section ".ext_zvar_attr", align 4{{$}}
// CHECK: @int_zvar = internal global i32 0, section ".int_zvar_attr", align 4{{$}}
// CHECK: define void @ext_fun() #0 section ".ext_fun_attr"
// CHECK: define void @ext_fun2() #0 section ".ext_fun2_attr"
// CHECK: define internal void @int_fun() #0 section ".int_fun_attr"
// CHECK: define internal void @int_fun2() #0 section ".int_fun2_attr"
//
// Function attributes should not include implicit-section-name.
// CHECK-NOT: attributes #0 = {{.*}}implicit-section-name
//
// No other attribute group should be present in the file.
// CHECK-NOT: attributes #1
