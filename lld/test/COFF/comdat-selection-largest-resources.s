# REQUIRES: x86

# A resource chunk in a losing LARGEST group must be removed before resource
# object classification and inclusion. Linking winner+loser in either order is
# observably equivalent to linking the winner alone.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /timestamp:0 \
# RUN:   /include:resource_group \
# RUN:   %t.large.obj /out:%t.winner.dll
# RUN: lld-link /dll /noentry /nodefaultlib /timestamp:0 \
# RUN:   /include:resource_group \
# RUN:   %t.small.obj %t.large.obj /out:%t.small-first.dll
# RUN: lld-link /dll /noentry /nodefaultlib /timestamp:0 \
# RUN:   /include:resource_group \
# RUN:   %t.large.obj %t.small.obj /out:%t.large-first.dll
# RUN: cmp %t.winner.dll %t.small-first.dll
# RUN: cmp %t.winner.dll %t.large-first.dll
# RUN: llvm-objdump -s %t.winner.dll | FileCheck %s

# CHECK: Contents of section .rsrc:
# CHECK: 4c415247 452d5245 534f5552 434500
# CHECK-NOT: 534d414c

#--- small.s
        .section .rsrc$01,"dr",largest,resource_group
        .globl resource_group
resource_group:
        .ascii "SMALL"

#--- large.s
        .section .rsrc$01,"dr",largest,resource_group
        .globl resource_group
resource_group:
        .asciz "LARGE-RESOURCE"
