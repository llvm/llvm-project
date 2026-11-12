# REQUIRES: x86
# RUN: llvm-mc %s -filetype=obj -o %t.obj -triple=x86_64-windows-msvc
# RUN: lld-link /entry:main /out:%t.exe /LARGEADDRESSAWARE:NO %t.obj
# RUN: llvm-readobj --coff-basereloc %t.exe | FileCheck %s --check-prefix=CHECKPASS

# This test case checks that the "ADDR32" relocation symbol is collected
# when linking a 64bit executable, and the output contains the HIGHLOW
# relocated symbol.

# Check that the HIGHLOW relocation base type is in the generated executable
# CHECKPASS: Format: COFF-x86-64
# CHECKPASS: Arch: x86_64
# CHECKPASS: AddressSize: 64bit
# CHECKPASS: BaseReloc
# CHECKPASS:   Entry {
# CHECKPASS-NEXT:     Type: HIGHLOW

    .text
    .def     main;
    .scl    2;
    .type   32;
    .endef
    .intel_syntax noprefix
    .globl  main
    .p2align        4, 0x90

main: # @main
    sub rsp, 40

    mov dword ptr [rip + arr + 24], 7

    mov eax, 1
    mov ecx, 20
    mov eax, dword ptr [rcx + 4 * rax + arr]

    ret

    .globl  arr
    .p2align        4
arr:
    .zero 40

