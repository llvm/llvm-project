## Test parsing of R (Return), r (FT_EXTERNAL_RETURN), and -1 hex (BR_ONLY)
## records in pre-aggregated profiles, plus error paths for invalid record
## types and malformed hex addresses. Closes e2e gaps that were previously
## only exercised by unit tests.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: link_fdata %s %t %t.preagg PREAGG

## Parse R, r, B-with-neg1, T-with-neg1, plus a regular F as a sanity check.
# RUN: perf2bolt %t -p %t.preagg --pa -o %t.fdata 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-PARSE
# CHECK-PARSE: read 6 aggregated brstack entries

## Error path: invalid record type letter.
# RUN: not perf2bolt %t -o %t.bad.fdata --pa \
# RUN:   -p %p/Inputs/pre-aggregated-bad-type.txt 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BAD-TYPE
# CHECK-BAD-TYPE: expected T, R, S, E, B, F, f or r

## Error path: malformed hex address.
# RUN: not perf2bolt %t -o %t.bad.fdata --pa \
# RUN:   -p %p/Inputs/pre-aggregated-bad-hex.txt 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BAD-HEX
# CHECK-BAD-HEX: expected hexadecimal number

  .text
  .globl main
  .type main, @function
main:
  pushq %rbp
  movq  %rsp, %rbp
  cmpl  $0, %edi
Lbr:
  je    Lfalse
Ltrue:
  movl  $1, %eax
  jmp   Lret
Lfalse:
  xorl  %eax, %eax
Lret:
  popq  %rbp
Lretins:
  retq
  .size main, .-main

## R: Return - branch in main returning to caller's continuation.
# PREAGG: R #Lretins# #Lretins# #Lbr# 5
## R with branch=0: falls back to FT_EXTERNAL_RETURN sentinel.
# PREAGG: R 0 #Lretins# #Lbr# 2
## r: fall-through after an external return.
# PREAGG: r #Lbr# #Ltrue# 7
## B with -1 (HEAD commit feature): parsed as BR_ONLY (overwritten anyway).
# PREAGG: B #Lbr# -1 1 0
## T with -1 as fall-through end: BR_ONLY sentinel preserved.
# PREAGG: T #Lbr# #Ltrue# -1 4
## F: regular fall-through sanity check.
# PREAGG: F #Ltrue# #Lret# 3
