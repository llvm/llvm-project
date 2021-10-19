# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hello.s -o %t.hello32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --no-entry --no-gc-sections --allow-undefined -fatal-warnings -o %t.wasm %t.o %t.hello32.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.section .x_sec,"",@
internal_x_seg_pad:
  # padding for provisioning value assertion
  .int32 0
  .size internal_x_seg_pad, 4
internal_x_seg:
  .int32 42
  .size internal_x_seg, 4

# internal cross segment subtraction
.section .foo,"",@
.globl  foo
foo:
  .int32 internal_x_seg - foo
  .size foo, 4
foo_addend:
  .int32 internal_x_seg - foo
  .size foo_addend, 4

# external cross segment subtraction
.section .bar,"",@
.globl  bar
bar:
  .int32 hello_str - bar
  .size bar, 4
bar_addend:
  .int32 hello_str - bar
  .size bar_addend, 4

# positive calc result
.section .fizz,"",@
.globl  fizz
fizz:
  .int32 far - fizz
  .size fizz, 4
fizz_addend:
  .int32 far - fizz
  .size fizz_addend, 4

.section .far,"",@
.globl  far
far:
  .int32 21
  .size far, 4

# CHECK:      - Type:            DATA
# CHECK-NEXT:    Segments:
# CHECK-NEXT:      - SectionOffset:   7
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1024
# CHECK-NEXT:        Content:         68656C6C6F0A00
# CHECK-NEXT:      - SectionOffset:   20
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1031
# CHECK-NEXT:        Content:         000000002A000000
# CHECK-NEXT:      - SectionOffset:   34
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1039
# CHECK-NEXT:        Content:         FCFFFFFFFCFFFFFF
# CHECK-NEXT:      - SectionOffset:   48
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1047
# CHECK-NEXT:        Content:         E9FFFFFFE9FFFFFF
# CHECK-NEXT:      - SectionOffset:   62
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1055
# CHECK-NEXT:        Content:         '0800000008000000'
# CHECK-NEXT:      - SectionOffset:   76
# CHECK-NEXT:        InitFlags:       0
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1063
# CHECK-NEXT:        Content:         '15000000'

