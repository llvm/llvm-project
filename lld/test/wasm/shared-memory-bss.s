# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -shared --shared-memory -o %t.so %t.o
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t.so | FileCheck %s
# RUN: obj2yaml %t.so | FileCheck %s --check-prefix=YAML

.section  .bss.foo,"",@
.globl  foo
.p2align  2
foo:
  .int32  0
  .size foo, 4

.section  .data.bar,"",@
.globl  bar
.p2align  2
bar:
  .int32  42
  .size bar, 4

.section  .custom_section.target_features,"",@
  .int8 2
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"

# Verify that there is only a single data segment and no bss
# in the binary:

#      YAML:  - Type:            DATA{{$}}
# YAML-NEXT:    Segments:
# YAML-NEXT:      - SectionOffset:   3
# YAML-NEXT:        InitFlags:       1
# YAML-NEXT:        Content:         2A000000
# YAML-NEXT:  - Type:            CUSTOM

# CHECK:      <__wasm_init_memory>:
# CHECK-NEXT:    .local i32
# CHECK-NEXT:            global.get      0
# CHECK-NEXT:            i32.const       8
# CHECK-NEXT:            i32.add
# CHECK-NEXT:            local.set       0
# CHECK-NEXT:            block
# CHECK-NEXT:            block
# CHECK-NEXT:            block
# CHECK-NEXT:            local.get       0
# CHECK-NEXT:            i32.const       0
# CHECK-NEXT:            i32.const       1
# CHECK-NEXT:            i32.atomic.rmw.cmpxchg  0
# CHECK-NEXT:            br_table        {0, 1, 2}       # 1: down to label1
# CHECK-NEXT:                                            # 2: down to label0
# CHECK-NEXT:            end

# Regular data gets initialized with memory.init

# CHECK-NEXT:            i32.const       0
# CHECK-NEXT:            global.get      0
# CHECK-NEXT:            i32.add
# CHECK-NEXT:            i32.const       0
# CHECK-NEXT:            i32.const       4
# CHECK-NEXT:            memory.init     0, 0

# BSS gets initialized with memory.fill

# CHECK-NEXT:            i32.const       4
# CHECK-NEXT:            global.get      0
# CHECK-NEXT:            i32.add
# CHECK-NEXT:            i32.const       0
# CHECK-NEXT:            i32.const       4
# CHECK-NEXT:            memory.fill     0
