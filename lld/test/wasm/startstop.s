# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/explicit-section.s -o %t2.o
# RUN: wasm-ld --export=get_start --export=get_end --export=foo --export=var1 %t.o %t2.o -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=ASM

.globl get_start
get_start:
    .functype   get_start () -> (i32)
    i32.const   __start_mysection
    end_function

.globl get_end
get_end:
    .functype   get_end () -> (i32)
    i32.const   __stop_mysection
    end_function

.globl _start
_start:
    .functype   _start () -> ()
    end_function

.section    mysection,"",@
.globl      foo
.p2align    2
foo:
    .int32      3
    .size       foo, 4

.globl      bar
.p2align    2
bar:
    .int32      4
    .size       bar, 4

#      CHECK:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   8
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:         Content:         03000000040000002A0000002B000000

#       ASM: 00000069 <get_start>:
# ASM-EMPTY:
#  ASM-NEXT:        6b:     i32.const 65536
#  ASM-NEXT:        71:     end

#       ASM: 00000072 <get_end>:
# ASM-EMPTY:
#  ASM-NEXT:        74:     i32.const 65552
#  ASM-NEXT:        7a:     end
