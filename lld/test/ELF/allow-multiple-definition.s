# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/allow-multiple-definition.s -o %t2
# RUN: not ld.lld %t1 %t2 -o /dev/null
# RUN: not ld.lld --allow-multiple-definition --no-allow-multiple-definition %t1 %t2 -o /dev/null
# RUN: ld.lld --allow-multiple-definition --fatal-warnings %t1 %t2 -o %t3
# RUN: ld.lld --allow-multiple-definition --fatal-warnings %t2 %t1 -o %t4
# RUN: llvm-objdump --no-print-imm-hex -d %t3 | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t4 | FileCheck --check-prefix=REVERT %s

# RUN: ld.lld -z muldefs --fatal-warnings  %t1 %t2 -o %t3
# RUN: ld.lld -z muldefs --fatal-warnings  %t2 %t1 -o %t4
# RUN: llvm-objdump --no-print-imm-hex -d %t3 | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t4 | FileCheck --check-prefix=REVERT %s

# inputs contain different constants for instruction movl.
# Tests below checks that order of files in command line
# affects on what symbol will be used.
# If flag allow-multiple-definition is enabled the first
# meet symbol should be used.

# CHECK: <_bar>:
# CHECK-NEXT:  movl   $1, %eax

# REVERT: <_bar>:
# REVERT-NEXT: movl   $2, %eax

.globl _bar
.type _bar, @function
_bar:
  mov $1, %eax

.globl _start
_start:
