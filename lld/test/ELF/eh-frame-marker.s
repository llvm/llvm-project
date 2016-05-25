// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld --eh-frame-hdr %t.o -o %t.so -shared
// We used to crash on this.
        .section .eh_frame
foo:
        .long 0
