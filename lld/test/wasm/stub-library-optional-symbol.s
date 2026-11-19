## Test stub library dependencies on optional linker-created symbols.
## See https://github.com/llvm/llvm-project/issues/180632

# RUN: split-file %s %t

# --- Case 1: __heap_base completely absent from object ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-absent.o %t/main-absent.s
# RUN: wasm-ld %t-absent.o %t/stub-heap.so -o %t-absent.wasm
# RUN: obj2yaml %t-absent.wasm | FileCheck %s --check-prefix=CHECK-ABSENT

# --- Case 2: __heap_base already Undefined in object ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-undef.o %t/main-undef.s
# RUN: wasm-ld %t-undef.o %t/stub-heap.so -o %t-undef.wasm
# RUN: obj2yaml %t-undef.wasm | FileCheck %s --check-prefix=CHECK-UNDEF

# --- Case 3: unknown stub dependency still errors ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %t/main-absent.s
# RUN: not wasm-ld %t.o %p/Inputs/libstub-missing-dep.so -o %t.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING

# --- Case 5: user-defined __data_end as DefinedData — linker must not override ---
# __heap_base cannot be assembled into a data section by llvm-mc; __data_end
# exercises the same materializeOptionalDataLayoutSymbol / ctx.sym.* path.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-user.o %t/main-user-data-end.s
# RUN: wasm-ld %t-user.o %t/stub-data-end.so -o %t-user.wasm
# RUN: obj2yaml %t-user.wasm | FileCheck %s --check-prefix=CHECK-USER

# --- Case 6: __memory_base optional global path (non-PIC) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-mem.o %t/main-absent.s
# RUN: wasm-ld %t-mem.o %t/stub-memory.so -o %t-mem.wasm
# RUN: obj2yaml %t-mem.wasm | FileCheck %s --check-prefix=CHECK-MEM

# --- Case 7: PIC __dso_handle stub dependency ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-dso.o %t/main-absent.s
# RUN: wasm-ld --experimental-pic -pie --import-memory %t-pic-dso.o %t/stub-dso.so -o %t-pic-dso.wasm
# RUN: obj2yaml %t-pic-dso.wasm | FileCheck %s --check-prefix=CHECK-PIC-DSO

# --- Case 8: PIC __wasm_first_page_end stub dependency ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-fpe.o %t/main-absent.s
# RUN: wasm-ld --experimental-pic -pie --import-memory %t-pic-fpe.o %t/stub-fpe.so -o %t-pic-fpe.wasm
# RUN: obj2yaml %t-pic-fpe.wasm | FileCheck %s --check-prefix=CHECK-PIC-FPE

# --- Case 9: PIC __heap_base stub dependency (must NOT be linker-created) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-pic-heap.o %t/main-absent.s
# RUN: not wasm-ld --experimental-pic -pie --import-memory %t-pic-heap.o %t/stub-heap.so -o %t-pic-heap.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-PIC-HEAP-FAIL

# --- Case 10: __tls_base stub dependency (distinct resolver path) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-tls.o %t/main-absent.s
# RUN: wasm-ld %t-tls.o %t/stub-tls.so -o %t-tls.wasm
# RUN: obj2yaml %t-tls.wasm | FileCheck %s --check-prefix=CHECK-TLS

# --- Case 11: __start_foo not yet supported by stub resolver (scope probe) ---
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t-start.o %t/main-startstop.s
# RUN: not wasm-ld %t-start.o %t/stub-startstop.so -o %t-start.wasm 2>&1 | FileCheck %s --check-prefix=CHECK-START-FAIL

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

# CHECK-MISSING: libstub-missing-dep.so: undefined symbol: missing_dep. Required by foo

# CHECK-USER:       - Name:            __data_end
# CHECK-USER-NEXT:    Kind:            GLOBAL
# CHECK-USER:         Content:         '63000000'

# CHECK-MEM:         Field:           foo_import
# CHECK-MEM:       - Name:            __memory_base
# CHECK-MEM-NEXT:    Kind:            GLOBAL

# CHECK-PIC-DSO:         Field:           foo_import
# CHECK-PIC-DSO:       - Name:            __dso_handle
# CHECK-PIC-DSO-NEXT:    Kind:            GLOBAL

# CHECK-PIC-FPE:         InitExpr:
# CHECK-PIC-FPE-NEXT:      Opcode:          I32_CONST
# CHECK-PIC-FPE-NEXT:      Value:           65536
# CHECK-PIC-FPE:       - Name:            __wasm_first_page_end
# CHECK-PIC-FPE-NEXT:    Kind:            GLOBAL

# CHECK-PIC-HEAP-FAIL: undefined symbol: __heap_base

# CHECK-TLS:       - Name:            __tls_base
# CHECK-TLS-NEXT:    Kind:            GLOBAL

# CHECK-START-FAIL: undefined symbol: __start_foo

#--- main-absent.s
.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function

#--- main-undef.s
.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    i32.const __heap_base@GOT
    drop
    call foo
    end_function

#--- main-user-data-end.s
.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function

.section .data,"",@
.globl __data_end
__data_end:
    .int32 99
    .size __data_end, 4

#--- main-startstop.s
.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function

.section foo,"",@
    .int32 42

#--- stub-data-end.so
#STUB
foo_import: __data_end

#--- stub-heap.so
#STUB
foo_import: __heap_base

#--- stub-dso.so
#STUB
foo_import: __dso_handle

#--- stub-fpe.so
#STUB
foo_import: __wasm_first_page_end

#--- stub-memory.so
#STUB
foo_import: __memory_base

#--- stub-tls.so
#STUB
foo_import: __tls_base

#--- stub-startstop.so
#STUB
foo_import: __start_foo
