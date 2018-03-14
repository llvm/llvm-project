; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -print-gc-sections -o %t1.wasm %t.o | FileCheck %s -check-prefix=PRINT-GC
; PRINT-GC: removing unused section {{.*}}:(unused_function)
; PRINT-GC-NOT: removing unused section {{.*}}:(used_function)
; PRINT-GC: removing unused section {{.*}}:(.data.unused_data)
; PRINT-GC-NOT: removing unused section {{.*}}:(.data.used_data)

target triple = "wasm32-unknown-unknown-wasm"

@unused_data = hidden global i64 1, align 4
@used_data = hidden global i32 2, align 4

define hidden i64 @unused_function() {
  %1 = load i64, i64* @unused_data, align 4
  ret i64 %1
}

define hidden i32 @used_function() {
  %1 = load i32, i32* @used_data, align 4
  ret i32 %1
}

define hidden void @_start() {
entry:
  call i32 @used_function()
  ret void
}

; RUN: obj2yaml %t1.wasm | FileCheck %s

; CHECK:        - Type:            TYPE
; CHECK-NEXT:     Signatures:      
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:      
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:      
; CHECK-NEXT:   - Type:            FUNCTION

; CHECK:        - Type:            DATA
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   7
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         '02000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            __wasm_call_ctors
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            used_function
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            _start
; CHECK-NEXT: ...

; RUN: wasm-ld -print-gc-sections --no-gc-sections -o %t1.no-gc.wasm %t.o
; RUN: obj2yaml %t1.no-gc.wasm | FileCheck %s -check-prefix=NO-GC

; NO-GC:        - Type:            TYPE
; NO-GC-NEXT:     Signatures:      
; NO-GC-NEXT:       - Index:           0
; NO-GC-NEXT:         ReturnType:      NORESULT
; NO-GC-NEXT:         ParamTypes:
; NO-GC-NEXT:       - Index:           1
; NO-GC-NEXT:         ReturnType:      I64
; NO-GC-NEXT:         ParamTypes:      
; NO-GC-NEXT:       - Index:           2
; NO-GC-NEXT:         ReturnType:      I32
; NO-GC-NEXT:         ParamTypes:      
; NO-GC-NEXT:   - Type:            FUNCTION

; NO-GC:        - Type:            DATA
; NO-GC-NEXT:     Segments:        
; NO-GC-NEXT:       - SectionOffset:   7
; NO-GC-NEXT:         MemoryIndex:     0
; NO-GC-NEXT:         Offset:          
; NO-GC-NEXT:           Opcode:          I32_CONST
; NO-GC-NEXT:           Value:           1024
; NO-GC-NEXT:         Content:         '010000000000000002000000'
; NO-GC-NEXT:   - Type:            CUSTOM
; NO-GC-NEXT:     Name:            name
; NO-GC-NEXT:     FunctionNames:
; NO-GC-NEXT:       - Index:           0
; NO-GC-NEXT:         Name:            __wasm_call_ctors
; NO-GC-NEXT:       - Index:           1
; NO-GC-NEXT:         Name:            unused_function
; NO-GC-NEXT:       - Index:           2
; NO-GC-NEXT:         Name:            used_function
; NO-GC-NEXT:       - Index:           3
; NO-GC-NEXT:         Name:            _start
; NO-GC-NEXT: ...

; RUN: not wasm-ld --gc-sections --relocatable -o %t1.no-gc.wasm %t.o 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
; CHECK-ERROR: error: -r and --gc-sections may not be used together
