// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
// RUN: ld.lld %t -z sort-thunksection -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=SORT
// RUN: not ld.lld -z nosort-thunksection %t -o /dev/null 2>&1 | FileCheck --check-prefix=NOSORT %s
// RUN: rm %t

// SORT-NOT: error: address assignment did not converge
// NOSORT: error: address assignment did not converge

// This test assumes pass limit of 30. Initially, only last thunk is out of range
// but as out of range thunks get promoted to long thunks, they force previous
// thunks to go out of range in the next pass. Without sorting the thunks,
// linker wouldn't converge. We only test absolute thunks which are the default
// thunks in the absence of --pic-thunks.

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

.rept NCALLS
  gen_call \+
.endr

// <Linker will create thunk section here>

.section .text.space, "ax", %progbits
// Position fn30 just past the 128 MiB (0x8000000) limit from created short thunk
// while all other functions are within 128MiB from their respective short thunks
// Absolute thunks are 16 bytes long.
.space 0x8000000 - (NCALLS - 1) * 16

.section .text.targets, "ax", %progbits

.rept NCALLS
  gen_target \+
.endr
