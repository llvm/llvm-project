# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/hidden.s -o %t2.o
# RUN: rm -f %t2.a
# RUN: llvm-ar rcs %t2.a %t2.o

# Test that symbols with hidden visibility are not exported, even with
# --export-dynamic
# RUN: wasm-ld --export-dynamic %t.o %t2.a -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# Test that symbols with default visibility are not exported without
# --export-dynamic
# RUN: wasm-ld %t.o %t2.a -o %t.nodef.wasm
# RUN: obj2yaml %t.nodef.wasm | FileCheck %s -check-prefix=NO-DEFAULT

.hidden objectHidden
objectHidden:
    .functype objectHidden () -> (i32)
    i32.const 0
    end_function

.globl objectHidden
.globl objectDefault
objectDefault:
    .functype objectDefault () -> (i32)
    i32.const 0
    end_function

.functype archiveHidden () -> (i32)
.functype archiveDefault () -> (i32)

.globl _start
_start:
    .functype _start () -> ()
    call objectHidden
    drop
    call objectDefault
    drop
    call archiveHidden
    drop
    call archiveDefault
    drop
    end_function


# CHECK:        - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            objectDefault
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            archiveDefault
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           4
# CHECK-NEXT:   - Type:


# NO-DEFAULT:        - Type:            EXPORT
# NO-DEFAULT-NEXT:     Exports:
# NO-DEFAULT-NEXT:       - Name:            memory
# NO-DEFAULT-NEXT:         Kind:            MEMORY
# NO-DEFAULT-NEXT:         Index:           0
# NO-DEFAULT-NEXT:       - Name:            _start
# NO-DEFAULT-NEXT:         Kind:            FUNCTION
# NO-DEFAULT-NEXT:         Index:           2
# NO-DEFAULT-NEXT:   - Type:
