; RUN: llc -filetype=obj %s -o %t.atomics.o -mattr=+atomics
; RUN: llc -filetype=obj %s -o %t.bulk-mem.o -mattr=+bulk-memory
; RUN: llc -filetype=obj %s -o %t.atomics.bulk-mem.o -mattr=+atomics,+bulk-memory

; atomics, shared memory => error
; RUN: not wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.o -o %t.atomics.wasm 2>&1 | FileCheck %s --check-prefix ERROR

; bulk memory, unshared memory => active segments
; RUN: wasm-ld -no-gc-sections --no-entry %t.bulk-mem.o -o %t.bulk-mem.wasm
; RUN: obj2yaml %t.bulk-mem.wasm | FileCheck %s --check-prefix ACTIVE

; atomics, bulk memory, shared memory => passive segments
; RUN: wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.o -o %t.atomics.bulk-mem.wasm
; RUN: obj2yaml %t.atomics.bulk-mem.wasm | FileCheck %s --check-prefixes PASSIVE

target triple = "wasm32-unknown-unknown"

@a = hidden global [6 x i8] c"hello\00", align 1
@b = hidden global [8 x i8] c"goodbye\00", align 1
@c = hidden global [10000 x i8] zeroinitializer, align 1
@d = hidden global i32 42, align 4

@e = private constant [9 x i8] c"constant\00", align 1
@f = private constant i8 43, align 4

; ERROR: 'bulk-memory' feature must be used in order to use shared memory

; ACTIVE-LABEL: - Type:            CODE
; ACTIVE-NEXT:    Functions:
; ACTIVE-NEXT:      - Index:           0
; ACTIVE-NEXT:        Locals:          []
; ACTIVE-NEXT:        Body:            0B
; ACTIVE-NEXT:  - Type:            DATA
; ACTIVE-NEXT:    Segments:
; ACTIVE-NEXT:      - SectionOffset:   7
; ACTIVE-NEXT:        InitFlags:       0
; ACTIVE-NEXT:        Offset:
; ACTIVE-NEXT:          Opcode:          I32_CONST
; ACTIVE-NEXT:          Value:           1024
; ACTIVE-NEXT:        Content:         636F6E7374616E74000000002B
; ACTIVE-NEXT:      - SectionOffset:   26
; ACTIVE-NEXT:        InitFlags:       0
; ACTIVE-NEXT:        Offset:
; ACTIVE-NEXT:          Opcode:          I32_CONST
; ACTIVE-NEXT:          Value:           1040
; ACTIVE-NEXT:        Content:         68656C6C6F00676F6F646279650000002A000000
; ACTIVE-NEXT:  - Type:            CUSTOM
; ACTIVE-NEXT:    Name:            name
; ACTIVE-NEXT:    FunctionNames:
; ACTIVE-NEXT:      - Index:           0
; ACTIVE-NEXT:        Name:            __wasm_call_ctors

; PASSIVE-LABEL: - Type:            START
; PASSIVE-NEXT:    StartFunction:   1
; PASSIVE-LABEL: - Type:            CODE
; PASSIVE-NEXT:    Functions:
; PASSIVE-NEXT:      - Index:           0
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            0B
; PASSIVE-NEXT:      - Index:           1
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            41B4D60041004101FE480200044041B4D6004101427FFE0102001A054180084100410DFC08000041900841004114FC08010041A40841004190CE00FC08020041B4D6004102FE17020041B4D600417FFE0002001A0BFC0900FC0901FC09020B
; PASSIVE-NEXT:  - Index:           2
; PASSIVE-NEXT:    Locals:          []
; PASSIVE-NEXT:    Body:            0B
; PASSIVE-NEXT:  - Type:            DATA
; PASSIVE-NEXT:    Segments:
; PASSIVE-NEXT:      - SectionOffset:   3
; PASSIVE-NEXT:        InitFlags:       1
; PASSIVE-NEXT:        Content:         636F6E7374616E74000000002B
; PASSIVE-NEXT:      - SectionOffset:   18
; PASSIVE-NEXT:        InitFlags:       1
; PASSIVE-NEXT:        Content:         68656C6C6F00676F6F646279650000002A000000
; PASSIVE-NEXT:  - Type:            CUSTOM
; PASSIVE-NEXT:    Name:            name
; PASSIVE-NEXT:    FunctionNames:
; PASSIVE-NEXT:      - Index:           0
; PASSIVE-NEXT:        Name:            __wasm_call_ctors
; PASSIVE-NEXT:      - Index:           1
; PASSIVE-NEXT:        Name:            __wasm_init_memory
; PASSIVE-NEXT:      - Index:           2
; PASSIVE-NEXT:        Name:            __wasm_init_tls
