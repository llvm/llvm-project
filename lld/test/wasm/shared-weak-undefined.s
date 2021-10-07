# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d %t.wasm | FileCheck %s -check-prefix=ASM

# Verify the weak undefined symbols are marked as such in the
# dylink section.

.weak weak_func
.functype weak_func () -> (i32)

.globl call_weak
call_weak:
# ASM: <call_weak>:
  .functype call_weak () -> (i32)
  call weak_func
# ASM:           10 80 80 80 80 00      call  0
  end_function
# ASM-NEXT:      0b                     end

#      CHECK: Sections:
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            dylink.0
# CHECK-NEXT:     MemorySize:      0
# CHECK-NEXT:     MemoryAlignment: 0
# CHECK-NEXT:     TableSize:       0
# CHECK-NEXT:     TableAlignment:  0
# CHECK-NEXT:     Needed:          []
# CHECK-NEXT:     ImportInfo:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           weak_func
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, UNDEFINED ]
