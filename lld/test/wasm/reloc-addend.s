# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -r -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=_start --no-show-raw-insn --no-leading-addr %t.wasm | FileCheck %s --check-prefixes DIS

.hidden foo
.hidden bar
.globl  foo
.globl  bar

# Similar to what would be generated from: `int foo[76]`
  .section  .bss.foo,"",@
  .p2align  4
foo:
  .skip 304
  .size foo, 304

# bar contains a pointer to the 16th element of foo, which happens to be 64
# bytes in.  This generates an addend of 64 which is the value at which signed
# and unsigned LEB encodes will differ.
  .section  .data.bar,"",@
  .p2align  2
bar:
  .int32  foo+64
  .size bar, 4

# Check that negative addends also work here
  .section  .data.negative_addend,"",@
  .p2align  2
negative_addend:
  .int32  foo-16
  .size negative_addend, 4

.globl _start
.section .text,"",@
_start:
  .functype _start () -> ()
  i32.const 0
  i32.load foo + 10
  drop
  i32.const 0
  i32.load foo - 10
  drop
  i32.const 0
  # This will underflow because i32.load (and the
  # corresponding relocation type) take an unsgiend (U32)
  # immediate.
  i32.load foo - 2048
  drop
  end_function

# CHECK:       - Type:            CODE
# CHECK-NEXT:    Relocations:
# CHECK-NEXT:      - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:        Offset:          0x7
# CHECK-NEXT:        Addend:          10
# CHECK-NEXT:      - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:        Offset:          0x11
# CHECK-NEXT:        Addend:          -10
# CHECK-NEXT:      - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:        Offset:          0x1B
# CHECK-NEXT:        Addend:          -2048

# CHECK:        - Type:            DATA
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x6
# CHECK-NEXT:         Addend:          64
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0xF
# CHECK-NEXT:         Addend:          -16

# DIS: <_start>:
# DIS-EMPTY:
# DIS-NEXT:    i32.const 0
# DIS-NEXT:    i32.load 26
# DIS-NEXT:    drop
# DIS-NEXT:    i32.const 0
# DIS-NEXT:    i32.load 6
# DIS-NEXT:    drop
# DIS-NEXT:    i32.const 0
# TODO(sbc): We should probably error here rather than allowing u32 to wrap
# DIS-NEXT:    i32.load 4294965264
# DIS-NEXT:    drop
# DIS-NEXT:    end
