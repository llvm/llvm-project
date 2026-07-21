// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o %t1.o
// RUN: ld.lld %t1.o -T a.lds -z sort-thunksection -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=SORT
// RUN: not ld.lld %t1.o -T a.lds -z nosort-thunksection -o /dev/null 2>&1 | FileCheck --check-prefix=NOSORT %s
// RUN: rm %t1.o

// SORT-NOT: error: address assignment did not converge
// NOSORT: error: address assignment did not converge

// This test assumes pass limit of 30. Initially, only last thunk is out of range
// but as out of range thunks get promoted to long thunks, they force previous
// thunks to go out of range in the next pass. Without sorting the thunks,
// linker wouldn't converge. We only test absolute thunks which are the default
// thunks in the absence of --pic-thunks. Absolute thunks occupies 16 bytes.

//--- a.lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  /* Position .text.targets such that initially only fn29 (the 30th call) is
     out of range */
  . = . + 0x8000000 - (29 * 16 + 4);
  .text.targets : { *(.text.targets) }
}

//--- a.s
.set NCALLS, 30

.macro gen_call i
  bl fn\i
  .space 12
.endm

.macro gen_target i
.global fn\i
fn\i:
  ret
  .space 12
.endm

.section .text, "ax", %progbits
.global _start
_start:

// Generates fn0, ... , fn29
.rept NCALLS
  gen_call \+
.endr

// <Linker will create thunk section here>

// Linker script positions .text.targets such that initially only fn29 is out
// of range.
.section .text.targets, "ax", %progbits

// Generates fn0, ... , fn29
.rept NCALLS
  gen_target \+
.endr
