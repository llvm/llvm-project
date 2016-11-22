# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld %t.o -o %t.so -shared 2>&1 | FileCheck %s

# CHECK: error: {{.*}}:(.data+0x0): undefined symbol 'hidden'
.global hidden
.hidden hidden

# CHECK: error: {{.*}}:(.data+0x8): undefined symbol 'internal'
.global internal
.internal internal

# CHECK: error: {{.*}}:(.data+0x10): undefined symbol 'protected'
.global protected
.protected protected

.section .data, "a"
 .quad hidden
 .quad internal
 .quad protected
