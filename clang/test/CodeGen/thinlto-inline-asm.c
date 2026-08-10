// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc19.11.0 -emit-llvm-bc \
// RUN:   -flto=thin -mllvm -x86-asm-syntax=intel -v \
// RUN:   -o %t.obj %s 2>&1 | FileCheck --check-prefix=CLANG %s
//
// RUN: llvm-lto2 dump-symtab %t.obj | FileCheck --check-prefix=SYMTAB %s

// Module-level inline asm is parsed with At&t syntax. Test that the
// -x86-asm-syntax flag does not affect this.

// CLANG-NOT: unknown token in expression
// SYMTAB: D------X foo
// SYMTAB: D------X bar

void foo(void) {}

asm(".globl bar      \n"
    "bar:            \n"
    "  xor %eax, %eax\n"
    "  ret           \n");
