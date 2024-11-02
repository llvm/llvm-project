# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --experimental-pic -shared -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d %t.wasm | FileCheck %s -check-prefix=ASM

# Run the same test but include a definition of ret32 in a library file.
# This verifies that LazySymbols (those found in library archives) are correctly
# demoted to undefined symbols in the final link when they are only weakly
# referenced.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.ret32.o %p/Inputs/ret32.s
# RUN: rm -f %T/libret32.a
# RUN: llvm-ar cru %T/libret32.a %t.ret32.o
# RUN: wasm-ld --experimental-pic -shared -o %t.ret32.wasm %t.o %T/libret32.a
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: llvm-objdump -d %t.wasm | FileCheck %s -check-prefix=ASM

# Verify the weak undefined symbols are marked as such in the
# dylink section.

.weak weak_func
.functype weak_func () -> (i32)
.weak ret32
.functype ret32 (f32) -> (i32)

.globl call_weak
call_weak:
# ASM: <call_weak>:
  .functype call_weak () -> (i32)
  call weak_func
# ASM:           10 80 80 80 80 00      call  0
  end_function
# ASM-NEXT:      0b                     end

# This function is defined in library archive, but since our reference to it
# is weak we don't expect this definition to be used.  Instead we expect it to
# act like an undefined reference and result in an imported function.
.globl call_weak_libfunc
call_weak_libfunc:
# ASM: <call_weak_libfunc>:
  .functype call_weak_libfunc () -> (i32)
  f32.const 1.0
  call ret32
# ASM:           10 81 80 80 80 00 call 1
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
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           ret32
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, UNDEFINED ]
