; RUN: llc -filetype=obj -o %t.o %s
; RUN: llc -filetype=obj %p/Inputs/explicit-section.ll -o %t2.o
; RUN: wasm-ld --export=get_start --export=get_end --export=foo --export=var1 %t.o %t2.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s
; RUN: llvm-objdump -d --no-show-raw-insn %t.wasm | FileCheck %s --check-prefix=ASM

target triple = "wasm32-unknown-unknown"

@foo = global i32 3, section "mysection", align 4
@bar = global i32 4, section "mysection", align 4

@__start_mysection = external global i8*
@__stop_mysection = external global i8*

define i8** @get_start() {
  ret i8** @__start_mysection
}

define i8** @get_end() {
  ret i8** @__stop_mysection
}

define void @_start()  {
entry:
  ret void
}

;      CHECK:   - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   7
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         03000000040000002A0000002B000000

;       ASM: 00000001 <get_start>:
; ASM-EMPTY:
;  ASM-NEXT:        3:     i32.const 1024
;  ASM-NEXT:        9:     end

;       ASM: 0000000a <get_end>:
; ASM-EMPTY:
;  ASM-NEXT:        c:     i32.const 1040
;  ASM-NEXT:       12:     end
