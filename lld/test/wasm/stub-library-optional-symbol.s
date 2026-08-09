## Test stub library dependencies on optional linker-created symbols.
## See https://github.com/llvm/llvm-project/issues/180632

# --- Case 1: __heap_base completely absent from object (exact #180632 bug) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-absent.o %S/Inputs/stub-optional-absent.s
# RUN: wasm-ld %t-absent.o %S/Inputs/libstub-heap-base.so -o %t-absent.wasm
# RUN: obj2yaml %t-absent.wasm | FileCheck %s --check-prefix=CHECK-ABSENT

# --- Case 2: __heap_base already Undefined in object (subtle v2 bug) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-undef.o %S/Inputs/stub-optional-undef.s
# RUN: wasm-ld %t-undef.o %S/Inputs/libstub-heap-base.so -o %t-undef.wasm
# RUN: obj2yaml %t-undef.wasm | FileCheck %s --check-prefix=CHECK-UNDEF

# --- Case 3: unknown stub dependency still errors ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %S/Inputs/stub-optional-absent.s
# RUN: not wasm-ld %t.o %p/Inputs/libstub-missing-dep.so -o %t.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING
# CHECK-MISSING: libstub-missing-dep.so: undefined symbol: missing_dep. Required by foo

# --- Case 5: user-defined __heap_base — linker must not override ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-user.o %S/Inputs/stub-optional-user-heap.s
# RUN: wasm-ld %t-user.o %S/Inputs/libstub-heap-base.so -o %t-user.wasm
# RUN: obj2yaml %t-user.wasm | FileCheck %s --check-prefix=CHECK-USER

# --- Case 6: __memory_base optional global path (non-PIC) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-mem.o %S/Inputs/stub-optional-memory.s
# RUN: wasm-ld %t-mem.o %S/Inputs/libstub-memory-base.so -o %t-mem.wasm
# RUN: obj2yaml %t-mem.wasm | FileCheck %s --check-prefix=CHECK-MEM

# CHECK-ABSENT:         Field:           foo_import
# CHECK-ABSENT:       - Type:            GLOBAL
# CHECK-ABSENT:         InitExpr:
# CHECK-ABSENT-NEXT:      Opcode:          I32_CONST
# CHECK-ABSENT-NEXT:      Value:           65536
# CHECK-ABSENT:       - Name:            __heap_base
# CHECK-ABSENT-NEXT:    Kind:            GLOBAL

# CHECK-UNDEF:         Field:           foo_import
# CHECK-UNDEF:       - Type:            GLOBAL
# CHECK-UNDEF:         InitExpr:
# CHECK-UNDEF-NEXT:      Opcode:          I32_CONST
# CHECK-UNDEF-NEXT:      Value:           65536
# CHECK-UNDEF:       - Name:            __heap_base
# CHECK-UNDEF-NEXT:    Kind:            GLOBAL

# CHECK-USER:       - Name:            __heap_base
# CHECK-USER-NEXT:    Kind:            GLOBAL

# CHECK-MEM:         Field:           foo_import
# CHECK-MEM:       - Name:            __memory_base
# CHECK-MEM-NEXT:    Kind:            GLOBAL

# --- Case 7: PIC __dso_handle stub dependency (must be created) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-dso.o %S/Inputs/stub-optional-absent.s
# RUN: wasm-ld --experimental-pic -pie --import-memory %t-pic-dso.o %S/Inputs/libstub-dso-handle.so -o %t-pic-dso.wasm
# RUN: obj2yaml %t-pic-dso.wasm | FileCheck %s --check-prefix=CHECK-PIC-DSO

# --- Case 8: PIC __wasm_first_page_end stub dependency (must be created) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-fpe.o %S/Inputs/stub-optional-absent.s
# RUN: wasm-ld --experimental-pic -pie --import-memory %t-pic-fpe.o %S/Inputs/libstub-first-page-end.so -o %t-pic-fpe.wasm
# RUN: obj2yaml %t-pic-fpe.wasm | FileCheck %s --check-prefix=CHECK-PIC-FPE

# --- Case 9: PIC __heap_base stub dependency (must NOT be linker-created) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-heap.o %S/Inputs/stub-optional-absent.s
# RUN: not wasm-ld --experimental-pic -pie --import-memory %t-pic-heap.o %S/Inputs/libstub-heap-base.so -o %t-pic-heap.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-PIC-HEAP-FAIL

# CHECK-PIC-DSO:         Field:           foo_import
# CHECK-PIC-DSO:       - Name:            __dso_handle
# CHECK-PIC-DSO-NEXT:    Kind:            GLOBAL

# CHECK-PIC-FPE:         InitExpr:
# CHECK-PIC-FPE-NEXT:      Opcode:          I32_CONST
# CHECK-PIC-FPE-NEXT:      Value:           65536
# CHECK-PIC-FPE:       - Name:            __wasm_first_page_end
# CHECK-PIC-FPE-NEXT:    Kind:            GLOBAL

# CHECK-PIC-HEAP-FAIL: undefined symbol: __heap_base
