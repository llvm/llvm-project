# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --entry=callWeakFuncs --print-gc-sections %t.o \
# RUN:     -o %t.wasm 2>&1 | FileCheck -check-prefix=CHECK-GC %s
# RUN: obj2yaml %t.wasm | FileCheck %s


# Check that calling an undefined weak function generates an appropriate stub
# that will fail at runtime with "unreachable".

.functype weakFunc1 () -> ()
.functype weakFunc2 () -> ()
.functype weakFunc3 (i32) -> ()
.functype weakFunc4 () -> ()

.globl  callWeakFuncs

callWeakFuncs:
  .functype callWeakFuncs () -> (i32)
  call  weakFunc1
  call  weakFunc2
  i32.const 2
  call  weakFunc3
  i32.const weakFunc1
  i32.const weakFunc4
  i32.add
  end_function

.weak weakFunc1
.weak weakFunc2
.weak weakFunc3
.weak weakFunc4

# CHECK-GC: removing unused section {{.*}}:(weakFunc4)

# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0, 1, 2 ]
# CHECK-NEXT:   - Type:            TABLE
# CHECK-NEXT:     Tables:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ElemType:        FUNCREF
# CHECK-NEXT:         Limits:
# CHECK-NEXT:           Flags:           [ HAS_MAX ]
# CHECK-NEXT:           Minimum:         0x1
# CHECK-NEXT:           Maximum:         0x1
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
# CHECK-NEXT:           Value:           66560
# CHECK-NEXT:   - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            callWeakFuncs
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            000B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            10808080800010818080800041021082808080004180808080004180808080006A0B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            'undefined_weak:weakFunc1'
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            'undefined_weak:weakFunc2'
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            'undefined_weak:weakFunc3'
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Name:            callWeakFuncs
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...
