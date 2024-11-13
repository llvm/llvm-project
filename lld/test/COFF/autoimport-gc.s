# REQUIRES: x86
# RUN: split-file %s %t.dir

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/lib.s -filetype=obj -o %t.dir/lib.obj
# RUN: lld-link -out:%t.dir/lib.dll -dll -entry:DllMainCRTStartup %t.dir/lib.obj -lldmingw -implib:%t.dir/lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %t.dir/main.s -filetype=obj -o %t.dir/main.obj
# RUN: lld-link -lldmingw -out:%t.dir/main.exe -entry:main %t.dir/main.obj %t.dir/lib.lib -opt:ref -debug:dwarf

#--- main.s
    .global main
    .section .text$main,"xr",one_only,main
main:
    ret

    .global other
    .section .text$other,"xr",one_only,other
other:
    movq .refptr.variable(%rip), %rax
    movl (%rax), %eax
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
