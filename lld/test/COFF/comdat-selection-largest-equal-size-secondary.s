# REQUIRES: x86
#
# Equal-sized LARGEST candidates keep the first selected leader. Secondary
# definitions from the losing equal-sized group must disappear, and providers
# hidden by those definitions must be recoverable independently of input order.
#
# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/a.s -o %t.a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/b.s -o %t.b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-a.s -o %t.provider-a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-b.s -o %t.provider-b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-lib -machine:amd64 -out:%t.providers.lib \
# RUN:   %t.provider-a.obj %t.provider-b.obj
#
# A wins the tie; only_b must come from its fallback provider.
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.a.obj %t.providers.lib %t.b.obj %t.root.obj \
# RUN:   /map:%t.ab.map /out:%t.ab.exe
# RUN: FileCheck %s --check-prefix=AB < %t.ab.map
# RUN: llvm-objdump -s %t.ab.exe | FileCheck %s --check-prefix=A-IMAGE \
# RUN:   --implicit-check-not=bbbbbbbb
#
# B wins the tie; only_a must come from its fallback provider.
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.b.obj %t.providers.lib %t.a.obj %t.root.obj \
# RUN:   /map:%t.ba.map /out:%t.ba.exe
# RUN: FileCheck %s --check-prefix=BA < %t.ba.map
# RUN: llvm-objdump -s %t.ba.exe | FileCheck %s --check-prefix=B-IMAGE \
# RUN:   --implicit-check-not=aaaaaaaa
#
# AB-DAG: only_a{{.*}}a.obj
# AB-DAG: only_b{{.*}}provider-b.obj
# BA-DAG: only_a{{.*}}provider-a.obj
# BA-DAG: only_b{{.*}}b.obj
#
# A-IMAGE-DAG: aaaaaaaa
# A-IMAGE-DAG: 22222222
# B-IMAGE-DAG: bbbbbbbb
# B-IMAGE-DAG: 11111111
#
#--- a.s
        .section .data$largest, "dw", largest, leader
        .globl leader
leader:
        .space 16, 0xaa
        .globl only_a
only_a:
        .long 0xaaaaaaaa
#
#--- b.s
        .section .data$largest, "dw", largest, leader
        .globl leader
leader:
        .space 16, 0xbb
        .globl only_b
only_b:
        .long 0xbbbbbbbb
#
#--- provider-a.s
        .section .rdata$provider_a, "dr"
        .globl only_a
only_a:
        .long 0x11111111
#
#--- provider-b.s
        .section .rdata$provider_b, "dr"
        .globl only_b
only_b:
        .long 0x22222222
#
#--- root.s
        .text
        .globl entry
entry:
        leaq leader(%rip), %rax
        movl only_a(%rip), %ecx
        movl only_b(%rip), %edx
        retq
