# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/common1.o %t/common1.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/common2.o %t/common2.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/strong.o %t/strong.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/weak.o %t/weak.s

# Test 1: Basic common symbol merging (max size and max alignment)
# RUN: wasm-ld --no-stack-first --global-base=1024 --export=foo --export=bar %t/common1.o %t/common2.o %t/main.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test 2: Strong defined symbol takes precedence over common symbol
# RUN: wasm-ld --no-stack-first --global-base=1024 --export=foo %t/common1.o %t/strong.o %t/main.o -o %t_strong.wasm
# RUN: obj2yaml %t_strong.wasm | FileCheck %s --check-prefix=STRONG

# Test 3: Common symbol takes precedence over weak defined symbol
# RUN: wasm-ld --no-stack-first --global-base=1024 --export=foo %t/common1.o %t/weak.o %t/main.o -o %t_weak.wasm
# RUN: obj2yaml %t_weak.wasm | FileCheck %s --check-prefix=WEAK

#--- common1.s
.comm foo, 4, 2
.comm bar, 8, 3

#--- common2.s
.comm foo, 8, 3
.comm bar, 4, 2

#--- strong.s
.globl foo
.section .data.foo,"",@
foo:
  .int32 42
.size foo, 4

#--- weak.s
.weak foo
.section .data.foo,"",@
foo:
  .int32 99
.size foo, 4

#--- main.s
.globl _start
_start:
  .functype _start () -> ()
  i32.const foo
  drop
  i32.const bar
  drop
  end_function

# CHECK:        - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66576
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1024
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1032

# STRONG:        - Type:            GLOBAL
# STRONG-NEXT:     Globals:
# STRONG-NEXT:       - Index:           0
# STRONG-NEXT:         Type:            I32
# STRONG-NEXT:         Mutable:         true
# STRONG-NEXT:         InitExpr:
# STRONG-NEXT:           Opcode:          I32_CONST
# STRONG-NEXT:           Value:           66576
# STRONG-NEXT:       - Index:           1
# STRONG-NEXT:         Type:            I32
# STRONG-NEXT:         Mutable:         false
# STRONG-NEXT:         InitExpr:
# STRONG-NEXT:           Opcode:          I32_CONST
# STRONG-NEXT:           Value:           1024
# STRONG:        - Type:            DATA
# STRONG-NEXT:     Segments:
# STRONG-NEXT:       - SectionOffset:   {{[0-9]+}}
# STRONG-NEXT:         InitFlags:       0
# STRONG-NEXT:         Offset:
# STRONG-NEXT:           Opcode:          I32_CONST
# STRONG-NEXT:           Value:           1024
# STRONG-NEXT:         Content:         2A000000

# WEAK-NOT:      - SectionOffset:   3
# WEAK-NOT:        Content:         '63000000'
# WEAK:        - Type:            GLOBAL
# WEAK-NEXT:     Globals:
# WEAK-NEXT:       - Index:           0
# WEAK-NEXT:         Type:            I32
# WEAK-NEXT:         Mutable:         true
# WEAK-NEXT:         InitExpr:
# WEAK-NEXT:           Opcode:          I32_CONST
# WEAK-NEXT:           Value:           66576
# WEAK-NEXT:       - Index:           1
# WEAK-NEXT:         Type:            I32
# WEAK-NEXT:         Mutable:         false
# WEAK-NEXT:         InitExpr:
# WEAK-NEXT:           Opcode:          I32_CONST
# WEAK-NEXT:           Value:           1024
