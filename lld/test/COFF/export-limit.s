# REQUIRES: x86
# RUN: %python %p/Inputs/def-many.py 65535 > %t-65535.def
# RUN: %python %p/Inputs/def-many.py 65536 > %t-65536.def
# RUN: llvm-mc -triple x86_64-win32 %s -filetype=obj -o %t.obj
# RUN: lld-link -dll -noentry %t.obj -out:%t.dll -def:%t-65535.def
# RUN: env LLD_IN_TEST=1 not lld-link -dll -noentry %t.obj -out:%t.dll -def:%t-65536.def 2>&1 | FileCheck %s

# CHECK: error: too many exported symbols (got 65536, max 65535)

        .text
        .globl f
f:
        ret
