# REQUIRES: x86

# RUN: llvm-mc -triple=i686-windows-gnu %s -defsym stdcall=0 -filetype=obj -o %t.obj

# RUN: lld-link -lldmingw -entry:main %t.obj -out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# RUN: llvm-mc -triple=i686-windows-gnu %s -defsym stdcall=1 -filetype=obj -o %t.stdcall.obj
# RUN: lld-link -lldmingw -entry:main %t.stdcall.obj -out:%t.stdcall.exe
# RUN: llvm-objdump -s %t.stdcall.exe | FileCheck %s

# Check that the .eh_frame comdat was included, even if it had no symbols,
# due to associativity with the symbol _foo.

# CHECK: Contents of section .eh_fram:
# CHECK:  403000 42

        .text
        .def            _main;
        .scl            2;
        .type           32;
        .endef
        .globl          _main
        .p2align        4, 0x90
_main:
.if stdcall==0
        call            _foo
.else
        call            _foo@0
.endif
        ret

        .section        .eh_frame$foo,"dr"
        .linkonce       discard
        .byte           0x42

.if stdcall==0
        .def            _foo;
.else
        .def            _foo@0;
.endif
        .scl            2;
        .type           32;
        .endef
.if stdcall==0
        .section        .text$foo,"xr",discard,_foo
        .globl          _foo
        .p2align        4
_foo:
.else
        .section        .text$foo,"xr",discard,_foo@0
        .globl          _foo@0
        .p2align        4
_foo@0:
.endif
        ret
