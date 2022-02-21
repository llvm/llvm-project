# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/call-indirect.s -o %t2.o
# RUN: wasm-ld --export-dynamic -o %t.wasm %t2.o %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

.globl foo
foo:
  .functype foo () -> (i32)
  i32.const 2
  end_function

.globl  _start
_start:
  .functype _start () -> ()
  i32.const 0
  i32.load indirect_func
  call_indirect () -> (i32)
  drop
  end_function

# Indirect function call where no function actually has this type.
# Ensures that the type entry is still created in this case.
.section .text,"",@
.globl call_ptr
call_ptr:
  .functype call_ptr (i32) -> ()
  i64.const 1
  local.get 0
  call_indirect (i64) -> (i64)
  drop
  end_function

.globl indirect_func
.section .data.indirect_func,"",@
indirect_func:
 .int32 foo
 .size indirect_func, 4

# CHECK:      !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 3, 1, 3, 4 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x3
# CHECK-NEXT:           Maximum:         0x3
# CHECK-NEXT:   - Type:            MEMORY
# CHECK-NEXT:     Memories:
# CHECK-NEXT:       - Minimum:         0x2
# CHECK-NEXT:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66576
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1032
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            bar
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            call_bar_indirect
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            foo
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:       - Name:            indirect_func
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            call_ptr
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           4
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1
# CHECK-NEXT:         Functions:       [ 0, 2 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            42010B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            410028028088808000118080808000001A410028028488808000118180808000001A0B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41020B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            410028028888808000118180808000001A0B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            42012000118280808000001A0B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:    7
# CHECK-NEXT:         InitFlags:        0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1024
# CHECK-NEXT:         Content:         '010000000200000002000000'
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            bar
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            call_bar_indirect
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            foo
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Name:            call_ptr
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT:     DataSegmentNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            .data
# CHECK-NEXT: ...
