# REQUIRES: x86

## common-symbol-alignment.elf contains common symbol with zero alignment.
# RUN: not ld.lld %S/Inputs/common-symbol-alignment.elf \
# RUN:   -o %t 2>&1 | FileCheck %s
# CHECK: common symbol 'bar' alignment is 0
