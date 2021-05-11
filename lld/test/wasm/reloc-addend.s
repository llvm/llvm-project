# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -r -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

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
