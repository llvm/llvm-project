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

.irp i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
  gen_call \i
.endr

// <Linker will create thunk section here>

.section .text.space, "ax", %progbits
// Position fn30 just past the 128 MiB (0x8000000) limit from created short thunk
// while all other functions are within 128MiB from their respective short thunks
// Absolute thunks are 16 bytes long.
.space 0x8000000 - 29 * 16

.section .text.targets, "ax", %progbits

.irp i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
  gen_target \i
.endr
