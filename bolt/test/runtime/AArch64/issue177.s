# This reproduces issue 177 from our github repo
# AARCH64_MOVW_UABS_G* relocations handling

# REQUIRES: system-linux

# RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0 --trap-old-code
# RUN: %t.bolt

.text
.align 4
.global test
.type test, %function
test:
mov x0, xzr
ret
.size test, .-test

.align 4
.global main
.type main, %function
main:
movz x0, #:abs_g3:test
movk x0, #:abs_g2_nc:test
movk x0, #:abs_g1_nc:test
movk x0, #:abs_g0_nc:test
br x0
.size main, .-main
