// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-elf -S %s -o - | FileCheck %s -check-prefix CHECK-ELF-DATA-SECTION

typedef struct __CFString *CFStringRef;
const CFStringRef one = (CFStringRef)__builtin___CFStringMakeConstantString("one");
const CFStringRef two = (CFStringRef)__builtin___CFStringMakeConstantString("\xef\xbf\xbd\x74\xef\xbf\xbd\x77\xef\xbf\xbd\x6f");

// CHECK-ELF-DATA-SECTION: .type .L.str,@object
// CHECK-ELF-DATA-SECTION: .section .rodata,"a",@progbits
// CHECK-ELF-DATA-SECTION: .L.str:
// CHECK-ELF-DATA-SECTION: .asciz "one"

// CHECK-ELF-DATA-SECTION: .type .L.str.1,@object
// CHECK-ELF-DATA-SECTION: .section .rodata,"a",@progbits
// CHECK-ELF-DATA-SECTION: .L.str.1:
// CHECK-ELF-DATA-SECTION: .short 65533
// CHECK-ELF-DATA-SECTION: .short 116
// CHECK-ELF-DATA-SECTION: .short 65533
// CHECK-ELF-DATA-SECTION: .short 119
// CHECK-ELF-DATA-SECTION: .short 65533
// CHECK-ELF-DATA-SECTION: .short 111
// CHECK-ELF-DATA-SECTION: .short 0
