## Check that --function-order can be combined with
## --reorder-functions algorithms to pin specific functions first
## while remaining functions are ordered by the selected algorithm.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q

## Create order file pinning func_b first, then func_a.
# RUN: echo "func_b" > %t.order
# RUN: echo "func_a" >> %t.order

## Test 1: Hybrid exec-count + order file.
## Order-file functions come first (func_b, func_a), then remaining sorted by
## execution count descending (func_c=100, func_d=50).
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-functions=exec-count \
# RUN:   --function-order=%t.order --generate-function-order=%t.genorder1 \
# RUN:   -o %t.null 2>&1 | FileCheck %s --check-prefix=CHECK-HYBRID
# RUN: FileCheck %s --input-file %t.genorder1 --check-prefix=CHECK-HYBRID-ORDER

# CHECK-HYBRID: BOLT-INFO: hybrid mode: functions from
# CHECK-HYBRID-SAME: order file will be pinned first,
# CHECK-HYBRID-SAME: remaining functions ordered by exec-count
# CHECK-HYBRID: BOLT-INFO: 2 functions pinned by order file

# CHECK-HYBRID-ORDER:      func_b
# CHECK-HYBRID-ORDER-NEXT: func_a
# CHECK-HYBRID-ORDER-NEXT: func_c
# CHECK-HYBRID-ORDER-NEXT: func_d

## Test 2: Standalone order file (no --reorder-functions).
## Should auto-default to --reorder-functions=user.
## Order-file functions come first, then remaining in original address order
## (func_d before func_c in the binary).
# RUN: llvm-bolt %t.exe --data %t.fdata \
# RUN:   --function-order=%t.order --generate-function-order=%t.genorder2 \
# RUN:   -o %t.null 2>&1 | FileCheck %s --check-prefix=CHECK-STANDALONE
# RUN: FileCheck %s --input-file %t.genorder2 \
# RUN:   --check-prefix=CHECK-STANDALONE-ORDER

# CHECK-STANDALONE: BOLT-INFO: --function-order specified
# CHECK-STANDALONE-SAME: without --reorder-functions,
# CHECK-STANDALONE-SAME: defaulting to --reorder-functions=user
# CHECK-STANDALONE: BOLT-INFO: 2 functions pinned by order file

# CHECK-STANDALONE-ORDER:      func_b
# CHECK-STANDALONE-ORDER-NEXT: func_a
# CHECK-STANDALONE-ORDER-NEXT: func_d
# CHECK-STANDALONE-ORDER-NEXT: func_c

## Test 3: User mode + order file. Same result as none + order file since
## RT_USER delegates entirely to the order file.
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-functions=user \
# RUN:   --function-order=%t.order --generate-function-order=%t.genorder3 \
# RUN:   -o %t.null 2>&1 | FileCheck %s --check-prefix=CHECK-USER
# RUN: FileCheck %s --input-file %t.genorder3 --check-prefix=CHECK-USER-ORDER

# CHECK-USER: BOLT-INFO: 2 functions pinned by order file

# CHECK-USER-ORDER:      func_b
# CHECK-USER-ORDER-NEXT: func_a
# CHECK-USER-ORDER-NEXT: func_d
# CHECK-USER-ORDER-NEXT: func_c

## Test 4: Explicit --reorder-functions=none + order file.
## Should warn and reset to --reorder-functions=user.
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-functions=none \
# RUN:   --function-order=%t.order --generate-function-order=%t.genorder4 \
# RUN:   -o %t.null 2>&1 | FileCheck %s --check-prefix=CHECK-RESET
# RUN: FileCheck %s --input-file %t.genorder4 --check-prefix=CHECK-RESET-ORDER

# CHECK-RESET: BOLT-WARNING: --reorder-functions=none
# CHECK-RESET-SAME: is incompatible with --function-order,
# CHECK-RESET-SAME: resetting to --reorder-functions=user
# CHECK-RESET: BOLT-INFO: 2 functions pinned by order file

# CHECK-RESET-ORDER:      func_b
# CHECK-RESET-ORDER-NEXT: func_a
# CHECK-RESET-ORDER-NEXT: func_d
# CHECK-RESET-ORDER-NEXT: func_c

## Test 5: Order file with a missing function.
## The nonexistent function is skipped with a warning, valid functions are still
## pinned correctly.
# RUN: echo "func_b" > %t.order_missing
# RUN: echo "nonexist" >> %t.order_missing
# RUN: echo "func_a" >> %t.order_missing
# RUN: llvm-bolt %t.exe --data %t.fdata --reorder-functions=exec-count \
# RUN:   --function-order=%t.order_missing \
# RUN:   --generate-function-order=%t.genorder5 -v=1 \
# RUN:   -o %t.null 2>&1 | FileCheck %s --check-prefix=CHECK-MISS
# RUN: FileCheck %s --input-file %t.genorder5 \
# RUN:   --check-prefix=CHECK-MISS-ORDER

# CHECK-MISS-DAG: can't find function for nonexist
# CHECK-MISS-DAG: can't find functions for 1 entries
# CHECK-MISS-DAG: 2 functions pinned by order file

# CHECK-MISS-ORDER:      func_b
# CHECK-MISS-ORDER-NEXT: func_a
# CHECK-MISS-ORDER-NEXT: func_c
# CHECK-MISS-ORDER-NEXT: func_d

  .text
  .align 4
  .globl main
  .type main, %function
main:
	.cfi_startproc
	bl func_a
	ret
	.cfi_endproc
.size main, .-main

  .globl func_a
  .type func_a, %function
func_a:
# FDATA: 0 [unknown] 0 1 func_a 0 0 10
	.cfi_startproc
	ret
	.cfi_endproc
.size func_a, .-func_a

  .globl func_b
  .type func_b, %function
func_b:
# FDATA: 0 [unknown] 0 1 func_b 0 0 5
	.cfi_startproc
	ret
	.cfi_endproc
.size func_b, .-func_b

## func_d is placed before func_c in the assembly to distinguish address-order
## results (func_d, func_c) from exec-count results (func_c, func_d).
  .globl func_d
  .type func_d, %function
func_d:
# FDATA: 0 [unknown] 0 1 func_d 0 0 50
	.cfi_startproc
	ret
	.cfi_endproc
.size func_d, .-func_d

  .globl func_c
  .type func_c, %function
func_c:
# FDATA: 0 [unknown] 0 1 func_c 0 0 100
	.cfi_startproc
	ret
	.cfi_endproc
.size func_c, .-func_c

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
