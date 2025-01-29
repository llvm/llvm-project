# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared2.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -soname=so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: not ld.lld -pie %t.o %t.so -o %t2 2>&1 | FileCheck -DFILE=%t %s --implicit-check-not=error:

# CHECK:      error: relocation R_AARCH64_AUTH_ABS64 cannot be used against symbol 'zed2'; recompile with -fPIC
# CHECK-NEXT: >>> defined in [[FILE]].so
# CHECK-NEXT: >>> referenced by [[FILE]].o:(.ro+0x0)

# CHECK:      error: relocation R_AARCH64_AUTH_ABS64 cannot be used against symbol 'bar2'; recompile with -fPIC
# CHECK:      error: relocation R_AARCH64_AUTH_ABS64 cannot be used against local symbol; recompile with -fPIC

foo:
.type foo, @function

.section .ro, "a"
.p2align 3
.quad zed2@AUTH(da,42)
.quad bar2@AUTH(ia,42)
.quad foo@AUTH(ia,42)
