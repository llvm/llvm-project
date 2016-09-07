# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# Simple symbol assignment within input section list. The '.' symbol
# is not location counter but offset from the beginning of output
# section .foo
# RUN: echo "SECTIONS { \
# RUN:          .foo : { \
# RUN:              begin_foo = .; \
# RUN:              PROVIDE(_begin_sec = .); \
# RUN:              *(.foo) \
# RUN:              end_foo = .; \
# RUN:              PROVIDE_HIDDEN(_end_sec = .); \
# RUN:              size_foo_1 = SIZEOF(.foo); \
# RUN:              . = ALIGN(0x1000); \
# RUN:              begin_bar = .; \
# RUN:              *(.bar) \
# RUN:              end_bar = .; \
# RUN:              size_foo_2 = SIZEOF(.foo); } \
# RUN:          size_foo_3 = SIZEOF(.foo); \
# RUN:          .eh_frame_hdr : { \
# RUN:             __eh_frame_hdr_start = .; \
# RUN:             __eh_frame_hdr_start2 = ALIGN(0x10); \
# RUN:             *(.eh_frame_hdr) \
# RUN:             __eh_frame_hdr_end = .; \
# RUN:             __eh_frame_hdr_end2 = ALIGN(0x10); } \
# RUN:       }" > %t.script
# RUN: ld.lld -o %t1 --eh-frame-hdr --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck --check-prefix=SIMPLE %s

# The script below contains symbols in the middle of .eh_frame_hdr section.
# We don't support this.
# RUN: echo "SECTIONS { \
# RUN:          .eh_frame_hdr : { \
# RUN:             PROVIDE_HIDDEN(_begin_sec = .); \
# RUN:             __eh_frame_hdr_start = .; \
# RUN:             *(.eh_frame_hdr) \
# RUN:             __eh_frame_hdr_end = .; \
# RUN:             *(.eh_frame_hdr) } \
# RUN:             PROVIDE_HIDDEN(_end_sec = .); \
# RUN:         }" > %t.script
# RUN: not ld.lld -o %t1 --eh-frame-hdr --script %t.script %t 2>&1 | FileCheck --check-prefix=ERROR %s

# Check that the following script is processed without errors
# RUN: echo "SECTIONS { \
# RUN:          .eh_frame_hdr : { \
# RUN:             PROVIDE_HIDDEN(_begin_sec = .); \
# RUN:             *(.eh_frame_hdr) \
# RUN:             *(.eh_frame_hdr) \
# RUN:             PROVIDE_HIDDEN(_end_sec = .); } \
# RUN:         }" > %t.script
# RUN: ld.lld -o %t1 --eh-frame-hdr --script %t.script %t

# SIMPLE:      0000000000000160         .foo    00000000 .hidden _end_sec
# SIMPLE:      0000000000000158         .foo    00000000 _begin_sec
# SIMPLE-NEXT: 0000000000000158         .foo    00000000 begin_foo
# SIMPLE-NEXT: 0000000000000160         .foo    00000000 end_foo
# SIMPLE-NEXT: 0000000000000008         .foo    00000000 size_foo_1
# SIMPLE-NEXT: 0000000000001000         .foo    00000000 begin_bar
# SIMPLE-NEXT: 0000000000001004         .foo    00000000 end_bar
# SIMPLE-NEXT: 0000000000000eac         .foo    00000000 size_foo_2
# SIMPLE-NEXT: 0000000000000eac         *ABS*   00000000 size_foo_3
# SIMPLE-NEXT: 0000000000001004         .eh_frame_hdr     00000000 __eh_frame_hdr_start
# SIMPLE-NEXT: 0000000000001010         .eh_frame_hdr     00000000 __eh_frame_hdr_start2
# SIMPLE-NEXT: 0000000000001018         .eh_frame_hdr     00000000 __eh_frame_hdr_end
# SIMPLE-NEXT: 0000000000001020         .eh_frame_hdr     00000000 __eh_frame_hdr_end2
# ERROR: section '.eh_frame_hdr' supports only start and end symbols

.global _start
_start:
 nop

.section .foo,"a"
 .quad 0

.section .bar,"a"
 .long 0

.section .dah,"ax",@progbits
 .cfi_startproc
 nop
 .cfi_endproc

.global _begin_sec, _end_sec
