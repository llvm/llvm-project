## Test that BOLT emits an absolute (TOC-independent) address materialization
## for PPC64 absolute call stubs, rather than a TOC/r2-relative sequence.
##
## Regression test: buildCallStubAbsolute previously used
##   addis r12, r2, sym@ha ; ld r12, sym@lo(r12)
## which requires a valid TOC in r2. The stub must instead build the target
## absolutely with lis/ori/sldi/oris/ori and must NOT reference r2.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple powerpc64le-unknown-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -e _start --emit-relocs
# RUN: llvm-bolt %t.exe -o %t.bolt -lite 2>&1 | FileCheck %s --check-prefix=BOLT
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=STUB

# BOLT: BOLT-INFO: Target architecture: powerpc64le
# BOLT: BOLT-INFO: enabling relocation mode

## The absolute call stub must materialize its target with no r2 dependence.
# STUB-LABEL: <__bolt_ppc_abs_call_stub.foo>:
# STUB:       lis     12
# STUB:       ori     12, 12
# STUB:       sldi    12, 12, 32
# STUB:       oris    12, 12
# STUB:       ori     12, 12
# STUB:       bctrl
# STUB-NOT:   addis   12, 2

        .text
        .abiversion 2
        .globl foo
        .type  foo, @function
foo:
        .localentry foo, 1
        blr
        .size foo, .-foo
        .globl _start
        .type  _start, @function
_start:
        .localentry _start, 1
        bl      foo
        nop
        li      0, 1
        li      3, 0
        sc
        .size _start, .-_start
