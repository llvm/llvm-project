; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.o -mattr=+atomics
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.bulk-mem.o -mattr=+bulk-memory
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.o -mattr=+atomics,+bulk-memory
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem64.o -mattr=+atomics,+bulk-memory
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.pic.o -relocation-model=pic -mattr=+atomics,+bulk-memory,+mutable-globals
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.pic-mem64.o -relocation-model=pic -mattr=+atomics,+bulk-memory,+mutable-globals

; atomics, shared memory => error
; RUN: not wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.o -o %t.atomics.wasm 2>&1 | FileCheck %s --check-prefix ERROR

; bulk memory, unshared memory => active segments
; RUN: wasm-ld -no-gc-sections --no-entry %t.bulk-mem.o -o %t.bulk-mem.wasm
; RUN: obj2yaml %t.bulk-mem.wasm | FileCheck %s --check-prefix ACTIVE

; atomics, bulk memory, shared memory => passive segments
; RUN: wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.o -o %t.atomics.bulk-mem.wasm
; RUN: obj2yaml %t.atomics.bulk-mem.wasm | FileCheck %s --check-prefixes PASSIVE,PASSIVE32

; Also test with wasm64
; RUN: wasm-ld -mwasm64 -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem64.o -o %t.atomics.bulk-mem64.wasm
; RUN: obj2yaml %t.atomics.bulk-mem64.wasm | FileCheck %s --check-prefixes PASSIVE,PASSIVE64

; Also test in combination with PIC/pie
; RUN: wasm-ld --experimental-pic -pie -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.pic.o -o %t.pic.wasm
; RUN: obj2yaml %t.pic.wasm | FileCheck %s --check-prefixes PASSIVE-PIC,PASSIVE32-PIC

; Also test in combination with PIC/pie + wasm64
; RUN: wasm-ld -mwasm64 --experimental-pic -pie -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.pic-mem64.o -o %t.pic-mem64.wasm
; RUN: obj2yaml %t.pic-mem64.wasm | FileCheck %s --check-prefixes PASSIVE-PIC,PASSIVE64-PIC

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
; PASSIVE-LABEL: - Type:            DATACOUNT
; PASSIVE-NEXT:    Count:           2
; PASSIVE-LABEL: - Type:            CODE
; PASSIVE-NEXT:    Functions:
; PASSIVE-NEXT:      - Index:           0
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            0B
; PASSIVE-NEXT:      - Index:           1
; PASSIVE-NEXT:        Locals:          []
; PASSIVE32-NEXT:        Body:            41B4D60041004101FE480200044041B4D6004101427FFE0102001A054180084100410DFC08000041900841004114FC08010041B4D6004102FE17020041B4D600417FFE0002001A0BFC0900FC09010B
; PASSIVE64-NEXT:        Body:            42B4D60041004101FE480200044042B4D6004101427FFE0102001A054280084100410DFC08000042900841004114FC08010042B4D6004102FE17020042B4D600417FFE0002001A0BFC0900FC09010B

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

;      PASSIVE-PIC:  - Type:            START
; PASSIVE-PIC-NEXT:    StartFunction:   2
; PASSIVE-PIC-NEXT:  - Type:            DATACOUNT
; PASSIVE-PIC-NEXT:    Count:           1
; PASSIVE-PIC-NEXT:  - Type:            CODE
; PASSIVE-PIC-NEXT:    Functions:
; PASSIVE-PIC-NEXT:      - Index:           0
; PASSIVE-PIC-NEXT:        Locals:          []
; PASSIVE-PIC-NEXT:        Body:            10010B
; PASSIVE-PIC-NEXT:      - Index:           1
; PASSIVE-PIC-NEXT:        Locals:          []
; PASSIVE-PIC-NEXT:        Body:            0B
; PASSIVE-PIC-NEXT:      - Index:           2
; PASSIVE-PIC-NEXT:        Locals:
; PASSIVE32-PIC-NEXT:          - Type:            I32
; PASSIVE64-PIC-NEXT:          - Type:            I64
; PASSIVE-PIC-NEXT:            Count:           1
; PASSIVE32-PIC-NEXT:        Body:            230141B4CE006A2100200041004101FE480200044020004101427FFE0102001A05410023016A410041B1CE00FC08000020004102FE1702002000417FFE0002001A0BFC09000B
; PASSIVE64-PIC-NEXT:        Body:            230142B4CE006A2100200041004101FE480200044020004101427FFE0102001A05420023016A410041B1CE00FC08000020004102FE1702002000417FFE0002001A0BFC09000B
; PASSIVE-PIC-NEXT:      - Index:           3
; PASSIVE-PIC-NEXT:        Locals:          []
; PASSIVE-PIC-NEXT:        Body:            0B
; PASSIVE-PIC-NEXT:  - Type:            DATA
; PASSIVE-PIC-NEXT:    Segments:
; PASSIVE-PIC-NEXT:      - SectionOffset:   4
; PASSIVE-PIC-NEXT:        InitFlags:       1

;      PASSIVE-PIC:  - Type:            CUSTOM
; PASSIVE-PIC-NEXT:    Name:            name
; PASSIVE-PIC-NEXT:    FunctionNames:
; PASSIVE-PIC-NEXT:      - Index:           0
; PASSIVE-PIC-NEXT:        Name:            __wasm_call_ctors
; PASSIVE-PIC-NEXT:      - Index:           1
; PASSIVE-PIC-NEXT:        Name:            __wasm_apply_relocs
; PASSIVE-PIC-NEXT:      - Index:           2
; PASSIVE-PIC-NEXT:        Name:            __wasm_init_memory
; PASSIVE-PIC-NEXT:      - Index:           3
; PASSIVE-PIC-NEXT:        Name:            __wasm_init_tls
