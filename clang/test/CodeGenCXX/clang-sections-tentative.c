// RUN: %clang_cc1 -x c -emit-llvm -triple arm-none-eabi -o - %s | FileCheck %s

// Test that section attributes are attached to C tentative definitions as per
// '#pragma clang section' directives.

#pragma clang section bss = ".bss.1"
int x; // bss.1

#pragma clang section bss = ""
int x; // stays in .bss.1
int y; // not assigned a section attribute
int z; // not assigned a section attribute

#pragma clang section bss = ".bss.2"
int x; // stays in .bss.1
int y; // .bss.2

// Test the same for `const` declarations.
#pragma clang section rodata = ".rodata.1"
const int cx; // rodata.1

#pragma clang section rodata = ""
const int cx; // stays in .rodata.1
const int cy; // not assigned a section attribute
const int cz; // not assigned a rodata section attribute

#pragma clang section rodata = ".rodata.2"
const int cx; // stays in .rodata.1
const int cy; // .rodata.2

// CHECK: @x ={{.*}} global i32 0, align 4 #0
// CHECK: @y ={{.*}} global i32 0, align 4 #1
// CHECK: @z ={{.*}} global i32 0, align 4
// CHECK: @cx ={{.*}} constant i32 0, align 4 #2
// CHECK: @cy ={{.*}} constant i32 0, align 4 #3
// CHECK: @cz ={{.*}} constant i32 0, align 4 #1

// CHECK: attributes #0 = { "bss-section"=".bss.1" }
// CHECK: attributes #1 = { "bss-section"=".bss.2" }
// CHECK: attributes #2 = { "bss-section"=".bss.2" "rodata-section"=".rodata.1" }
// CHECK: attributes #3 = { "bss-section"=".bss.2" "rodata-section"=".rodata.2" }
