# REQUIRES: x86

# RUN: echo -e "EXPORTS\nvariable1 DATA\nvariable2 DATA" > %t-lib.def
# RUN: llvm-dlltool -m i386:x86-64 -d %t-lib.def -D lib.dll -l %t-lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw -out:%t.exe -entry:main %t.obj %t-lib.lib -verbose 2>&1 | FileCheck %s

# CHECK-NOT: runtime pseudo relocation {{.*}} against symbol variable1
# CHECK: warning: runtime pseudo relocation in {{.*}}.obj against symbol variable2 is too narrow (only 32 bits wide); this can fail at runtime depending on memory layout
# CHECK-NOT: runtime pseudo relocation {{.*}} against symbol variable1

    .global main
    .text
main:
    movq .refptr.variable1(%rip), %rax
    movl (%rax), %eax
    movl variable2(%rip), %ecx
    addl %ecx, %eax
    ret
    .global _pei386_runtime_relocator
_pei386_runtime_relocator:
    ret

    .section .rdata$.refptr.variable1,"dr",discard,.refptr.variable1
    .global .refptr.variable1
.refptr.variable1:
    .quad variable1

relocs:
    .quad __RUNTIME_PSEUDO_RELOC_LIST__
    .quad __RUNTIME_PSEUDO_RELOC_LIST_END__
