# REQUIRES: x86

# Test loaded archive members containing competing largest COMDATs. Unique force
# symbols ensure that both members are extracted in the lazy-archive cases.
# /wholearchive exercises the same selection when every member is loaded.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/reference-small.s -o %t.reference-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/reference-provider.s -o %t.reference-provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/reference-root.s -o %t.reference-root.obj
# RUN: llvm-lib -machine:amd64 -out:%t.small-large.lib %t.small.obj %t.large.obj
# RUN: llvm-lib -machine:amd64 -out:%t.large-small.lib %t.large.obj %t.small.obj
# RUN: llvm-lib -machine:amd64 -out:%t.reference-provider.lib \
# RUN:   %t.reference-provider.obj

# RUN: lld-link /opt:noref /include:force_small /include:force_large \
# RUN:   /dll /noentry /nodefaultlib %t.small-large.lib \
# RUN:   /out:%t.lazy.small-large.exe
# RUN: llvm-objdump -s %t.lazy.small-large.exe | FileCheck %s

# RUN: lld-link /opt:noref /include:force_large /include:force_small \
# RUN:   /dll /noentry /nodefaultlib %t.large-small.lib \
# RUN:   /out:%t.lazy.large-small.exe
# RUN: llvm-objdump -s %t.lazy.large-small.exe | FileCheck %s

# RUN: lld-link /opt:noref /dll /noentry /nodefaultlib \
# RUN:   /wholearchive:%t.small-large.lib /out:%t.whole.small-large.exe
# RUN: llvm-objdump -s %t.whole.small-large.exe | FileCheck %s

# RUN: lld-link /opt:noref /dll /noentry /nodefaultlib \
# RUN:   /wholearchive:%t.large-small.lib /out:%t.whole.large-small.exe
# RUN: llvm-objdump -s %t.whole.large-small.exe | FileCheck %s

# Archive extraction is intentionally monotonic. A reference from the current
# winner may extract a member which then replaces that winner with a larger
# COMDAT; the already extracted member remains part of the link.
# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.reference-small.obj %t.reference-provider.lib \
# RUN:   %t.reference-root.obj /out:%t.reference-extraction.exe
# RUN: llvm-objdump -s %t.reference-extraction.exe | \
# RUN:   FileCheck --check-prefix=REFERENCE-EXTRACTION %s

# CHECK: Contents of section .text:
# CHECK-NEXT:  180001000 44444444

# REFERENCE-EXTRACTION: Contents of section .text:
# REFERENCE-EXTRACTION: 44444444 44444444 44444444 44444444
# REFERENCE-EXTRACTION: Contents of section .rdata:
# REFERENCE-EXTRACTION: aaaaaaaa

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .section .rdata$force, "dr"
        .globl force_small
force_small:
        .byte 1

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444

        .section .rdata$force, "dr"
        .globl force_large
force_large:
        .byte 4

#--- reference-small.s
        .section .text$largest, "xr", largest, reference_leader
        .globl reference_leader
reference_leader:
        callq reference_provider
        retq

#--- reference-provider.s
        .section .text$largest, "xr", largest, reference_leader
        .globl reference_leader
reference_leader:
        .space 32, 0x44

        .section .rdata$provider, "dr"
        .globl reference_provider
reference_provider:
        .long 0xaaaaaaaa

#--- reference-root.s
        .text
        .globl entry
entry:
        leaq reference_leader(%rip), %rax
        retq
