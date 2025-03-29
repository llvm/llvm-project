# REQUIRES: x86
# RUN: split-file %s %t.dir && cd %t.dir

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows test.s -o test.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows drectve.s -o drectve.obj

# RUN: env LLD_IN_TEST=1 not lld-link -out:out.dll test.obj -dll -entry: 2>&1 | FileCheck %s
# RUN: env LLD_IN_TEST=1 not lld-link -out:out.dll test.obj -dll drectve.obj 2>&1 | FileCheck %s

# CHECK: error: missing entry point symbol name

#--- test.s
        .text
        .globl func
func:
        ret

#--- drectve.s
        .section .drectve, "yn"
        .ascii " -entry:"
