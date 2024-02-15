# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -soname=so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: not ld.lld -pie %t.o %t.so -o %t2 2>&1 | FileCheck -DFILE=%t %s

# CHECK:      error: relocation R_AARCH64_AUTH_ABS64 against symbol 'zed2' in read-only section
# CHECK-NEXT: defined in [[FILE]].so
# CHECK-NEXT: referenced by [[FILE]].o:(.test+0x0)

# CHECK:      error: relocation R_AARCH64_AUTH_ABS64 against symbol 'bar2' in read-only section
# CHECK-NEXT: defined in [[FILE]].so
# CHECK-NEXT: referenced by [[FILE]].o:(.test+0x8)

.section .test, "a"
.p2align 3
.quad zed2@AUTH(da,42)
.quad bar2@AUTH(ia,42)
