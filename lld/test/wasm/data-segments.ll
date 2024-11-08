; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.o -mattr=+atomics,-bulk-memory
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.bulk-mem.o -mattr=+bulk-memory
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o %t.bulk-mem64.o -mattr=+bulk-memory
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.o -mattr=+atomics,+bulk-memory
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem64.o -mattr=+atomics,+bulk-memory
; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.pic.o -relocation-model=pic -mattr=+atomics,+bulk-memory,+mutable-globals
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o %t.atomics.bulk-mem.pic-mem64.o -relocation-model=pic -mattr=+atomics,+bulk-memory,+mutable-globals

; atomics, shared memory => error
; RUN: not wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.o -o %t.atomics.wasm 2>&1 | FileCheck %s --check-prefix ERROR

; bulk memory, unshared memory => active segments
; RUN: wasm-ld -no-gc-sections --no-entry %t.bulk-mem.o -o %t.bulk-mem.wasm
; RUN: obj2yaml %t.bulk-mem.wasm | FileCheck %s --check-prefixes ACTIVE,ACTIVE32

; bulk memory, unshared memory, wasm64 => active segments
; RUN: wasm-ld -mwasm64 -no-gc-sections --no-entry %t.bulk-mem64.o -o %t.bulk-mem64.wasm
; RUN: obj2yaml %t.bulk-mem64.wasm | FileCheck %s --check-prefixes ACTIVE,ACTIVE64

; atomics, bulk memory, shared memory => passive segments
; RUN: wasm-ld -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.o -o %t.atomics.bulk-mem.wasm
; RUN: obj2yaml %t.atomics.bulk-mem.wasm | FileCheck %s --check-prefix PASSIVE
; RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors,__wasm_init_memory --no-show-raw-insn --no-leading-addr %t.atomics.bulk-mem.wasm | FileCheck %s --check-prefixes DIS,NOPIC-DIS -DPTR=i32

; atomics, bulk memory, shared memory, wasm64 => passive segments
; RUN: wasm-ld -mwasm64 -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem64.o -o %t.atomics.bulk-mem64.wasm
; RUN: obj2yaml %t.atomics.bulk-mem64.wasm | FileCheck %s --check-prefix PASSIVE
; RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors,__wasm_init_memory --no-show-raw-insn --no-leading-addr %t.atomics.bulk-mem64.wasm | FileCheck %s --check-prefixes DIS,NOPIC-DIS -DPTR=i64

; Also test in combination with PIC/pie
; RUN: wasm-ld --experimental-pic -pie -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.pic.o -o %t.pic.wasm
; RUN: obj2yaml %t.pic.wasm | FileCheck %s --check-prefixes PASSIVE-PIC,PASSIVE32-PIC
; RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors,__wasm_apply_data_relocs,__wasm_init_memory --no-show-raw-insn --no-leading-addr %t.pic.wasm | FileCheck %s --check-prefixes DIS,PIC-DIS -DPTR=i32

; Also test in combination with PIC/pie + wasm64
; RUN: wasm-ld -mwasm64 --experimental-pic -pie -no-gc-sections --no-entry --shared-memory --max-memory=131072 %t.atomics.bulk-mem.pic-mem64.o -o %t.pic-mem64.wasm
; RUN: obj2yaml %t.pic-mem64.wasm | FileCheck %s --check-prefixes PASSIVE-PIC,PASSIVE64-PIC
; RUN: llvm-objdump --disassemble-symbols=__wasm_call_ctors,__wasm_apply_data_relocs,__wasm_init_memory --no-show-raw-insn --no-leading-addr %t.pic-mem64.wasm | FileCheck %s --check-prefixes DIS,PIC-DIS -DPTR=i64

@a = hidden global [6 x i8] c"hello\00", align 1
@b = hidden global [8 x i8] c"goodbye\00", align 1
@c = hidden global [10000 x i8] zeroinitializer, align 1
@d = hidden global i32 42, align 4

@e = private constant [9 x i8] c"constant\00", align 1
@f = private constant i8 43, align 4

@g = thread_local global i32 99, align 4

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
; ACTIVE32-NEXT:        Opcode:          I32_CONST
; ACTIVE64-NEXT:        Opcode:          I64_CONST
; ACTIVE-NEXT:          Value:           1024
; ACTIVE-NEXT:        Content:         636F6E7374616E74000000002B
; ACTIVE-NEXT:      - SectionOffset:   26
; ACTIVE-NEXT:        InitFlags:       0
; ACTIVE-NEXT:        Offset:
; ACTIVE32-NEXT:        Opcode:          I32_CONST
; ACTIVE64-NEXT:        Opcode:          I64_CONST
; ACTIVE-NEXT:          Value:           1040
; ACTIVE-NEXT:        Content:         68656C6C6F00676F6F646279650000002A000000
; ACTIVE-NEXT:  - Type:            CUSTOM
; ACTIVE-NEXT:    Name:            name
; ACTIVE-NEXT:    FunctionNames:
; ACTIVE-NEXT:      - Index:           0
; ACTIVE-NEXT:        Name:            __wasm_call_ctors

; PASSIVE-LABEL: - Type:            START
; PASSIVE-NEXT:    StartFunction:   2
; PASSIVE-LABEL: - Type:            DATACOUNT
; PASSIVE-NEXT:    Count:           3
; PASSIVE-LABEL: - Type:            CODE
; PASSIVE-NEXT:    Functions:
; PASSIVE-NEXT:      - Index:           0
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            0B
; PASSIVE-NEXT:      - Index:           1
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            {{.*}}
; PASSIVE-NEXT:      - Index:           2
; PASSIVE-NEXT:        Locals:          []
; PASSIVE-NEXT:        Body:            {{.*}}
; PASSIVE-NEXT:  - Type:            DATA
; PASSIVE-NEXT:    Segments:
; PASSIVE-NEXT:      - SectionOffset:   3
; PASSIVE-NEXT:        InitFlags:       1
; PASSIVE-NEXT:        Content:         '63000000'
; PASSIVE-NEXT:      - SectionOffset:   9
; PASSIVE-NEXT:        InitFlags:       1
; PASSIVE-NEXT:        Content:         636F6E7374616E74000000002B
; PASSIVE-NEXT:      - SectionOffset:   24
; PASSIVE-NEXT:        InitFlags:       1
; PASSIVE-NEXT:        Content:         68656C6C6F00676F6F646279650000002A000000
; PASSIVE-NEXT:  - Type:            CUSTOM
; PASSIVE-NEXT:    Name:            name
; PASSIVE-NEXT:    FunctionNames:
; PASSIVE-NEXT:      - Index:           0
; PASSIVE-NEXT:        Name:            __wasm_call_ctors
; PASSIVE-NEXT:      - Index:           1
; PASSIVE-NEXT:        Name:            __wasm_init_tls
; PASSIVE-NEXT:      - Index:           2
; PASSIVE-NEXT:        Name:            __wasm_init_memory

;      PASSIVE-PIC:  - Type:            START
; PASSIVE-PIC-NEXT:    StartFunction:   2
; PASSIVE-PIC-NEXT:  - Type:            DATACOUNT
; PASSIVE-PIC-NEXT:    Count:           3
; PASSIVE-PIC-NEXT:  - Type:            CODE
; PASSIVE-PIC-NEXT:    Functions:
; PASSIVE-PIC-NEXT:      - Index:           0
; PASSIVE-PIC-NEXT:        Locals:          []
; PASSIVE-PIC-NEXT:        Body:            0B
; PASSIVE-PIC-NEXT:      - Index:           1
; PASSIVE-PIC-NEXT:        Locals:          []
; PASSIVE-PIC-NEXT:        Body:            {{.*}}
; PASSIVE-PIC-NEXT:      - Index:           2
; PASSIVE-PIC-NEXT:        Locals:
; PASSIVE32-PIC-NEXT:          - Type:            I32
; PASSIVE64-PIC-NEXT:          - Type:            I64
; PASSIVE-PIC-NEXT:              Count:           2
; PASSIVE-PIC-NEXT:        Body:            {{.*}}
; PASSIVE-PIC-NEXT:  - Type:            DATA
; PASSIVE-PIC-NEXT:    Segments:
; PASSIVE-PIC-NEXT:      - SectionOffset:   3
; PASSIVE-PIC-NEXT:        InitFlags:       1
; PASSIVE-PIC-NEXT:        Content:         '63000000'
; PASSIVE-PIC-NEXT:      - SectionOffset:   9
; PASSIVE-PIC-NEXT:        InitFlags:       1
; PASSIVE-PIC-NEXT:        Content:         636F6E7374616E74000000002B
; PASSIVE-PIC-NEXT:      - SectionOffset:   24
; PASSIVE-PIC-NEXT:        InitFlags:       1
; PASSIVE-PIC-NEXT:        Content:         68656C6C6F00676F6F646279650000002A000000
; PASSIVE-PIC-NEXT:  - Type:            CUSTOM
; PASSIVE-PIC-NEXT:    Name:            name
; PASSIVE-PIC-NEXT:    FunctionNames:
; PASSIVE-PIC-NEXT:      - Index:           0
; PASSIVE-PIC-NEXT:        Name:            __wasm_call_ctors
; PASSIVE-PIC-NEXT:      - Index:           1
; PASSIVE-PIC-NEXT:        Name:            __wasm_init_tls
; PASSIVE-PIC-NEXT:      - Index:           2
; PASSIVE-PIC-NEXT:        Name:            __wasm_init_memory

; no data relocations.
; DIS-LABEL:       <__wasm_call_ctors>:
; DIS-EMPTY:
; DIS-NEXT:        end

; PIC-DIS-EMPTY:

; DIS-LABEL:       <__wasm_init_memory>:

; PIC-DIS:           .local [[PTR]]
; PIC-DIS-NEXT:      global.get      1
; PIC-DIS-NEXT:      [[PTR]].const   10040
; PIC-DIS-NEXT:      [[PTR]].add
; PIC-DIS-NEXT:      local.set       0

; DIS:               block
; DIS-NEXT:           block
; DIS-NEXT:            block

; NOPIC-DIS-NEXT:       [[PTR]].const   11064
; PIC-DIS-NEXT:         local.get       0

; DIS-NEXT:             i32.const       0
; DIS-NEXT:             i32.const       1
; DIS-NEXT:             i32.atomic.rmw.cmpxchg  0
; DIS-NEXT:             br_table        {0, 1, 2}      # 1:     down to label1
; DIS-NEXT:                                            # 2:     down to label0
; DIS-NEXT:            end

; NOPIC-DIS-NEXT:      [[PTR]].const   1024
; NOPIC-DIS-NEXT:      [[PTR]].const   1024
; NOPIC-DIS-NEXT:      global.set      1
; PIC-DIS-NEXT:        [[PTR]].const   0
; PIC-DIS-NEXT:        global.get      1
; PIC-DIS-NEXT:        [[PTR]].add
; PIC-DIS-NEXT:        local.tee       1
; PIC-DIS-NEXT:        global.set      {{\d*}}
; PIC-DIS-NEXT:        local.get       1
; DIS-NEXT:            i32.const       0
; DIS-NEXT:            i32.const       4
; DIS-NEXT:            memory.init  0, 0

; NOPIC-DIS-NEXT:      [[PTR]].const   1028
; PIC-DIS-NEXT:        [[PTR]].const   4
; PIC-DIS-NEXT:        global.get      1
; PIC-DIS-NEXT:        [[PTR]].add

; DIS-NEXT:            i32.const       0
; DIS-NEXT:            i32.const       13
; DIS-NEXT:            memory.init     1, 0

; NOPIC-DIS-NEXT:      [[PTR]].const   1044
; PIC-DIS-NEXT:        [[PTR]].const   20
; PIC-DIS-NEXT:        global.get      1
; PIC-DIS-NEXT:        [[PTR]].add

; DIS-NEXT:            i32.const       0
; DIS-NEXT:            i32.const       20
; DIS-NEXT:            memory.init     2, 0
; NOPIC-DIS-NEXT:      [[PTR]].const   1064
; PIC-DIS-NEXT:        [[PTR]].const   40
; PIC-DIS-NEXT:        global.get      1
; PIC-DIS-NEXT:        [[PTR]].add
; DIS-NEXT:            i32.const       0
; DIS-NEXT:            [[PTR]].const   10000
; DIS-NEXT:            memory.fill     0

; NOPIC-DIS-NEXT:      [[PTR]].const   11064
; PIC-DIS-NEXT:        local.get       0

; DIS-NEXT:            i32.const       2
; DIS-NEXT:            i32.atomic.store        0

; NOPIC-DIS-NEXT:      [[PTR]].const   11064
; PIC-DIS-NEXT:        local.get       0

; DIS-NEXT:            i32.const       -1
; DIS-NEXT:            memory.atomic.notify    0
; DIS-NEXT:            drop
; DIS-NEXT:            br              1               # 1:     down to label1
; DIS-NEXT:           end

; NOPIC-DIS-NEXT:     [[PTR]].const   11064
; PIC-DIS-NEXT:       local.get       0

; DIS-NEXT:           i32.const       1
; DIS-NEXT:           i64.const       -1
; DIS-NEXT:           memory.atomic.wait32    0
; DIS-NEXT:           drop
; DIS-NEXT:          end
; DIS-NEXT:          data.drop       1
; DIS-NEXT:          data.drop       2
; DIS-NEXT:         end
