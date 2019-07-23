# REQUIRES: x86

# RUN: echo -e ".section .tls,\"dw\"\n .byte 0xaa\n .section .tls\$ZZZ,\"dw\"\n .byte 0xff\n .globl _tls_index\n .data\n _tls_index:\n .int 0" > %t.tlssup.s
# RUN: llvm-mc -triple=x86_64-windows-gnu %t.tlssup.s -filetype=obj -o %t.tlssup.o
# RUN: llvm-mc -triple=x86_64-windows-gnu %s -filetype=obj -o %t.main.o

# RUN: lld-link -lldmingw -entry:main %t.main.o %t.tlssup.o -out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# Check that .tls$$foo is sorted after the start marker (aa) and before the
# end marker (ff).

# CHECK: Contents of section .tls:
# CHECK:  140004000 aabbff

        .text
        .globl          main
main:
        movl            _tls_index(%rip), %eax
        movq            %gs:88, %rcx
        movq            (%rcx,%rax,8), %rax
        movb            foo@SECREL32(%rax), %al
        ret

        .section        .tls$$foo,"dw"
        .linkonce       discard
foo:
        .byte           0xbb
