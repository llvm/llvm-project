// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-linux         -S -o - %s | FileCheck %s --check-prefix=ASM
// Actually, any ELF target would do
// REQUIRES: x86-registered-target

#pragma clang section bss = "B$$" data = "d@t@" rodata = "r0d@t@"

const int a = 1;
const int *f() { return &a; }

int init();
const int b = init();

int c = 2;

int d = init();

int e;

// LLVM: @_ZL1a = internal constant i32 1, align 4 #[[#A:]]
// LLVM: @_ZL1b = internal global i32 0, align 4 #[[#A]]
// LLVM: @c = {{.*}}global i32 2, align 4 #[[#A]]
// LLVM: @d = {{.*}}global i32 0, align 4 #[[#A]]
// LLVM: @e = {{.*}}global i32 0, align 4 #[[#A]]

// LLVM: attributes #[[#A]] = { "bss-section"="B$$" "data-section"="d@t@" "rodata-section"="r0d@t@" }

// ASM:       .section "r0d@t@","a",@progbits
// ASM-NOT:   .section
// ASM-LABEL: _ZL1a:
// ASM-NEXT:  .long 1

// ASM:       .section "B$$","aw",@nobits
// ASM-NOT:   .section
// ASM-LABEL: _ZL1b:
// ASM-NEXT: .long 0

// ASM:       .section "d@t@","aw",@progbits
// ASM-NOT:   .section
// ASM-LABEL: c:
// ASM:       .long 2

// ASM:       .section "B$$","aw",@nobits
// ASM-NOT:   .section
// ASM-LABEL: d:
// ASM:       .long 0

// ASM-NOT:   .section
// ASM-LABEL: e:
// ASM        .long 0
