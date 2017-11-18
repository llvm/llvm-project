; RUN: llc -filetype=obj %p/Inputs/call-indirect.ll -o %t2.o
; RUN: llc -filetype=obj %s -o %t.o
; RUN: lld -flavor wasm -o %t.wasm %t2.o %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; bitcode generated from the following C code:
; int foo(void) { return 1; }
; int (*indirect_func)(void) = &foo;
; void _start(void) { indirect_func(); }

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

@indirect_func = hidden local_unnamed_addr global i32 ()* @foo, align 4

; Function Attrs: norecurse nounwind readnone
define hidden i32 @foo() #0 {
entry:
  ret i32 1
}

; Function Attrs: nounwind
define hidden void @_start() local_unnamed_addr #1 {
entry:
  %0 = load i32 ()*, i32 ()** @indirect_func, align 4
  %call = tail call i32 %0() #2
  ret void
}

; CHECK:      !WASM
; CHECK-NEXT: FileHeader:      
; CHECK-NEXT:   Version:         0x00000001
; CHECK-NEXT: Sections:        
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:      
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:      
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ReturnType:      NORESULT
; CHECK-NEXT:         ParamTypes:      
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 1, 0, 1 ]
; CHECK-NEXT:   - Type:            TABLE
; CHECK-NEXT:     Tables:          
; CHECK-NEXT:       - ElemType:        ANYFUNC
; CHECK-NEXT:         Limits:          
; CHECK-NEXT:           Flags:           0x00000001
; CHECK-NEXT:           Initial:         0x00000003
; CHECK-NEXT:           Maximum:         0x00000003
; CHECK-NEXT:   - Type:            MEMORY
; CHECK-NEXT:     Memories:        
; CHECK-NEXT:       - Initial:         0x00000002
; CHECK-NEXT:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:         
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66576
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            bar
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            call_bar_indirect
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            _start
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           3
; CHECK:        - Type:            ELEM
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 0, 2 ]
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Functions:       
; CHECK:            - Locals:          
; CHECK:            - Locals:          
; CHECK:            - Locals:          
; CHECK:        - Type:            DATA
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:    7
; CHECK-NEXT:         MemoryIndex:      0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024
; CHECK-NEXT:         Content:         '0100000002000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        8
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            bar
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            call_bar_indirect
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            _start
