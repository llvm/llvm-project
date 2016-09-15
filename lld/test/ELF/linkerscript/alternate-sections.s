# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { abc : { *(foo) *(bar) *(zed) } }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o -shared
# RUN: llvm-readobj -s -section-data %t | FileCheck %s

# This test shows an oddity in lld. When a linker script alternates among
# different types of output section in the same command, the sections are
# reordered.
# In this test we go from regular, to merge and back to regular. The reason
# for the reordering is that we need two create two output sections and
# one cannot be in the middle of another.
# If this ever becomes a problem, some options would be:
# * Adding an extra layer in between input section and output sections (Chunk).
#   With that this example would have 3 chunks, but only one output section.
#   This would unfortunately complicate the non-script case too.
# * Just create three output sections.
# * If having three output sections causes problem, have linkerscript specific
#   code to write the section table and section indexes. That way we could
#   keep 3 sections internally but not expose that.

# CHECK: Name: abc
# CHECK: 0000: 01000000 00000000 02000000 00000000  |
# CHECK: Name: abc
# CHECK: 0000: 61626331 323300                      |abc123.|

        .section foo, "a"
        .quad 1

        .section bar,"aMS",@progbits,1
        .asciz  "abc123"

        .section zed, "a"
        .quad 2
