# REQUIRES: x86
# RUN: split-file %s %t.dir

## We've got references to variable both in a .refptr and in .debug_info.
## The .debug_info section should be discardable, so no pseudo relocations
## need to be created in it. The .refptr section should be elimiated
## and redirected to __imp_variable instead, so we shouldn't need to
## create any runtime pseudo relocations. Thus, test that we can link
## successfully with -runtime-pseudo-reloc:no, while keeping the
## debug info.

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/lib.s -filetype=obj -o %t.dir/lib.obj
# RUN: lld-link -out:%t.dir/lib.dll -dll -entry:DllMainCRTStartup %t.dir/lib.obj -lldmingw -implib:%t.dir/lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/main.s -filetype=obj -o %t.dir/main.obj
# RUN: lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/lib.lib -opt:noref -debug:dwarf -runtime-pseudo-reloc:no

#--- main.s
    .global main
    .text
main:
    movq .refptr.variable(%rip), %rax
    ret

    .section .rdata$.refptr.variable,"dr",discard,.refptr.variable
    .global .refptr.variable
.refptr.variable:
    .quad   variable

    .section .debug_info
    .long 1
    .quad variable
    .long 2

#--- lib.s
    .global variable
    .global DllMainCRTStartup
    .text
DllMainCRTStartup:
    ret
    .data
variable:
    .long 42
