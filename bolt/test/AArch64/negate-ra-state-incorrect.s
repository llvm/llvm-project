# This test checks that MarkRAStates pass ignores functions with
# malformed .cfi_negate_ra_state sequences in the input binary.

# The cases checked are:
#   - extra .cfi_negate_ra_state in Signed state: checked in foo,
#   - extra .cfi_negate_ra_state in Unsigned state: checked in bar,
#   - missing .cfi_negate_ra_state from PSign or PAuth instructions: checked in baz.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.exe.bolt --no-threads | FileCheck %s --check-prefix=CHECK-BOLT

# CHECK-BOLT: BOLT-INFO: inconsistent RAStates in function foo: ptr authenticating inst encountered in Unsigned RA state
# CHECK-BOLT: BOLT-INFO: inconsistent RAStates in function bar: ptr signing inst encountered in Signed RA state
# CHECK-BOLT: BOLT-INFO: inconsistent RAStates in function baz: ptr sign/auth inst without .cfi_negate_ra_state

# Check that the incorrect functions got ignored, so they are not in the new .text section
# RUN: llvm-objdump %t.exe.bolt -d -j .text | FileCheck %s --check-prefix=CHECK-OBJDUMP
# CHECK-OBJDUMP-NOT: <foo>:
# CHECK-OBJDUMP-NOT: <bar>:
# CHECK-OBJDUMP-NOT: <baz>:


  .text
  .globl  foo
  .p2align        2
  .type   foo,@function
foo:
  .cfi_startproc
  hint    #25
  .cfi_negate_ra_state
  mov x1, #0
  .cfi_negate_ra_state        // Incorrect CFI in signed state
  hint    #29
  .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   foo, .-foo

  .text
  .globl  bar
  .p2align        2
  .type   bar,@function
bar:
  .cfi_startproc
  mov x1, #0
  .cfi_negate_ra_state      // Incorrect CFI in unsigned state
  hint    #25
  .cfi_negate_ra_state
  mov x1, #0
  hint    #29
  .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   bar, .-bar

  .text
  .globl  baz
  .p2align        2
  .type   baz,@function
baz:
  .cfi_startproc
  mov x1, #0
  hint    #25
  .cfi_negate_ra_state
  mov x1, #0
  hint    #29
                            // Missing .cfi_negate_ra_state
  ret
  .cfi_endproc
  .size   baz, .-baz

  .global _start
  .type _start, %function
_start:
  b foo
  b bar
  b baz
