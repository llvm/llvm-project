# REQUIRES: x86

# This test ensures that we're following the MSVC symbol searching behvior described in:
# https://learn.microsoft.com/en-us/cpp/build/reference/link-input-files?view=msvc-170
# "Object files on the command line are processed in the order they appear on the command line.
# Libraries are searched in command line order as well, with the following caveat: Symbols that
# are unresolved when bringing in an object file from a library are searched for in that library
# first, and then the following libraries from the command line and /DEFAULTLIB (Specify default
# library) directives, and then to any libraries at the beginning of the command line."

# RUN: echo -e ".intel_syntax noprefix\n.globl libfunc\n.text\nlibfunc:\nmov eax, 1\nret\n.section .drectve\n.ascii \"/EXPORT:libfunc\"" > %t.lib.s
# RUN: llvm-mc -triple=x86_64-pc-windows-msvc %t.lib.s -filetype=obj -o %t.lib.o
# RUN: lld-link -dll -out:%t.lib.dll -entry:libfunc %t.lib.o -implib:%t.lib.dll.a

# RUN: echo -e ".globl helper\n.text\nhelper:\ncall libfunc\nret" > %t.helper1.s
# RUN: echo -e ".intel_syntax noprefix\n.globl libfunc\n.text\nlibfunc:\nxor eax, eax\nret" > %t.helper2.s
# RUN: llvm-mc -triple=x86_64-pc-windows-msvc %t.helper1.s -filetype=obj -o %t.helper1.o
# RUN: llvm-mc -triple=x86_64-pc-windows-msvc %t.helper2.s -filetype=obj -o %t.helper2.o

# RUN: llvm-ar rcs %t.helper.a %t.helper1.o %t.helper2.o

# RUN: llvm-mc -triple=x86_64-pc-windows-msvc %s -filetype=obj -o %t.main.o

# Simulate a setup, where two libraries provide the same function;
# %t.lib.dll.a is a pure import library which provides a import symbol "libfunc".
# %t.helper.a is a static library which contains "helper1" and "helper2".
#
# helper1 contains an undefined reference to libfunc. helper2 contains an
# implementation of libfunc.
#
# First %t.main.o is processed and pushes a undefined symbol 'helper'.
# Then %t.lib.dll.a is processed a pushes the lazy archive symbol 'libfunc' in the symbol table.
# Then comes %t.helper.a and it pushes 'helper' and 'libfunc' as lazy symbols. Then 'helper' is
# resolved and that pushes 'libfunc' as a undefined symbol. That pulls on %t.helper.a(%t.helper2.o)
# which contains the 'libfunc' symbol, resolving it. This is illustrative of the MSVC library searching
# behavior which starts with the current library object which requested the unresolved symbol.
# RUN: lld-link -out:%t.main.exe -entry:main %t.main.o %t.lib.dll.a %t.helper.a
# RUN: llvm-objdump --no-print-imm-hex -d %t.main.exe | FileCheck --check-prefix=LIB %s

# In this case, the symbol in %t.helper.a(%t.helper2.o) is still considered first.
# RUN: lld-link -out:%t.main.exe -entry:main %t.main.o %t.helper.a %t.lib.dll.a
# RUN: llvm-objdump --no-print-imm-hex -d %t.main.exe | FileCheck --check-prefix=LIB %s

# In this test we're defining libfunc in a third library that comes after all the others. The symbol should be pulled
# now from that third library.
# RUN: llvm-ar rcs %t.helper1.a %t.helper1.o
# RUN: llvm-ar rcs %t.helper2.a %t.helper2.o
# RUN: lld-link -out:%t.main.exe -entry:main %t.main.o %t.lib.dll.a %t.helper1.a %t.helper2.a
# RUN: llvm-objdump --no-print-imm-hex -d %t.main.exe | FileCheck --check-prefix=LIB %s

# LIB: 140001000 <.text>:
# LIB: 140001000: e8 03 00 00 00                   callq   0x140001008 <.text+0x8>
# LIB: 140001008: e8 03 00 00 00                   callq   0x140001010 <.text+0x10>
# LIB: 140001010: 31 c0                            xorl    %eax, %eax

# In this last test, we should pick up the import symbol from %t.lib.dll.a since it isn't defined anywhere else.
# RUN: lld-link -out:%t.main.exe -entry:main %t.main.o %t.lib.dll.a %t.helper1.a
# RUN: llvm-objdump --no-print-imm-hex -d %t.main.exe | FileCheck --check-prefix=LIB-IMP %s

# LIB-IMP: 140001000 <.text>:
# LIB-IMP: 140001010: ff 25 22 10 00 00            jmpq    *4130(%rip)

    .globl main
    .text
main:
    call helper
    ret
