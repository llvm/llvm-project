# Verify that the `no_strip` attribute on undefined symbols is preserved
# as long as one of the undefined references is `no_strip`.
# The indirect function table is created by the linker in the case that
# the __indirect_function_table symbol is live (i.e. marked as no_strip).
# This test verifies that the undefined symbol is NO_STRIP, even if the first
# undefined reference is not NO_STRIP.
#
# RUN: split-file %s %t
# RUN: llvm-mc -mattr=+reference-types -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -mattr=+reference-types -filetype=obj -triple=wasm32-unknown-unknown -o %t/other.o %t/other.s
#
# RUN: wasm-ld %t/main.o %t/other.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s
#
# RUN: wasm-ld %t/other.o %t/main.o -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s

#--- main.s
.globl __indirect_function_table
.tabletype __indirect_function_table, funcref
.no_dead_strip __indirect_function_table

.globl _start
_start:
  .functype _start () -> ()
  end_function

#--- other.s
# This file contains a reference to __indirect_function_table that is not
# marked as `.no_dead_strip`.
.globl __indirect_function_table
.tabletype __indirect_function_table, funcref

# CHECK:       - Type:            TABLE
# CHECK-NEXT:    Tables:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        ElemType:        FUNCREF
# CHECK-NEXT:        Limits:
# CHECK-NEXT:          Flags:           [ HAS_MAX ]
# CHECK-NEXT:          Minimum:         0x1
# CHECK-NEXT:          Maximum:         0x1
