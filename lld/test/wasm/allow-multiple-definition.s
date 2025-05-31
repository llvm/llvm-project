# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/allow-multiple-definition.s -o %t2
# RUN: not wasm-ld %t1 %t2 -o /dev/null
# RUN: not wasm-ld --allow-multiple-definition --no-allow-multiple-definition %t1 %t2 -o /dev/null
# RUN: wasm-ld --allow-multiple-definition --fatal-warnings %t1 %t2 -o %t3
# RUN: wasm-ld --allow-multiple-definition --fatal-warnings %t2 %t1 -o %t4
# RUN: llvm-objdump --no-print-imm-hex -d %t3 | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t4 | FileCheck --check-prefix=REVERT %s

# RUN: wasm-ld --noinhibit-exec %t2 %t1 -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# WARN: warning: duplicate symbol: foo

# RUN: wasm-ld -z muldefs --fatal-warnings  %t1 %t2 -o %t3
# RUN: wasm-ld -z muldefs --fatal-warnings  %t2 %t1 -o %t4
# RUN: llvm-objdump --no-print-imm-hex -d %t3 | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t4 | FileCheck --check-prefix=REVERT %s

# CHECK:  i32.const 0
# REVERT:  i32.const 1

# inputs contain different constants for function foo return.
# Tests below checks that order of files in command line
# affects on what symbol will be used.
# If flag allow-multiple-definition is enabled the first
# meet symbol should be used.

  .hidden foo
  .globl  foo
foo:
  .functype foo () -> (i32)
  i32.const 0
  end_function

  .globl _start
_start:
  .functype  _start () -> (i32)
  call foo
  end_function
