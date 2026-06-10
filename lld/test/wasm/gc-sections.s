# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: yaml2obj %S/Inputs/globals.yaml -o %t_globals.o
# RUN: wasm-ld -print-gc-sections -o %t1.wasm %t.o %t_globals.o | \
# RUN:     FileCheck %s -check-prefix=PRINT-GC
# PRINT-GC: removing unused section {{.*}}:(unused_function)
# PRINT-GC-NOT: removing unused section {{.*}}:(used_function)
# PRINT-GC: removing unused section {{.*}}:(.data.unused_data)
# PRINT-GC-NOT: removing unused section {{.*}}:(.data.used_data)
# PRINT-GC: removing unused section {{.*}}:(unused_global)
# PRINT-GC-NOT: removing unused section {{.*}}:(used_global)

.functype use_global () -> (i64)

.globl unused_function
unused_function:
  .functype unused_function (i64) -> (i64)
  i32.const 0
  i64.load  unused_data
  end_function

.globl used_function
used_function:
  .functype used_function () -> (i32)
  i32.const 0
  i32.load  used_data
  end_function

.globl _start
_start:
  .functype _start () -> ()
  call  used_function
  drop
  call use_global
  drop
  end_function

.globl unused_data
.section .data.unused_data,"",@
.p2align 3
unused_data:
  .int64 1
  .size unused_data, 8

.globl used_data
.section .data.used_data,"",@
.p2align 2
used_data:
  .int32 2
  .size used_data, 4

# RUN: obj2yaml %t1.wasm | FileCheck %s

# CHECK:        - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:     []
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I64
# CHECK-NEXT:   - Type:            FUNCTION

# CHECK:        - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I64
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I64_CONST
# CHECK-NEXT:           Value:           456

# CHECK:        - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   8
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           65536
# CHECK-NEXT:         Content:         '02000000'
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            used_function
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            use_global
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            used_global
# CHECK-NEXT:     DataSegmentNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            .data
# CHECK-NEXT: ...

# RUN: wasm-ld -print-gc-sections --no-gc-sections -o %t1.no-gc.wasm \
# RUN:     %t.o %t_globals.o
# RUN: obj2yaml %t1.no-gc.wasm | FileCheck %s -check-prefix=NO-GC

# NO-GC:        - Type:            TYPE
# NO-GC-NEXT:     Signatures:
# NO-GC-NEXT:       - Index:           0
# NO-GC-NEXT:         ParamTypes:      []
# NO-GC-NEXT:         ReturnTypes:     []
# NO-GC-NEXT:       - Index:           1
# NO-GC-NEXT:         ParamTypes:
# NO-GC-NEXT:           - I64
# NO-GC-NEXT:         ReturnTypes:
# NO-GC-NEXT:           - I64
# NO-GC-NEXT:       - Index:           2
# NO-GC-NEXT:         ParamTypes:      []
# NO-GC-NEXT:         ReturnTypes:
# NO-GC-NEXT:           - I32
# NO-GC-NEXT:       - Index:           3
# NO-GC-NEXT:         ParamTypes:      []
# NO-GC-NEXT:         ReturnTypes:
# NO-GC-NEXT:           - I64
# NO-GC-NEXT:   - Type:            FUNCTION

# NO-GC:        - Type:            GLOBAL
# NO-GC-NEXT:     Globals:
# NO-GC-NEXT:       - Index:           0
# NO-GC-NEXT:         Type:            I32
# NO-GC-NEXT:         Mutable:         true
# NO-GC-NEXT:         InitExpr:
# NO-GC-NEXT:           Opcode:          I32_CONST
# NO-GC-NEXT:           Value:           65536
# NO-GC-NEXT:       - Index:       1
# NO-GC-NEXT:         Type:        I64
# NO-GC-NEXT:         Mutable:     true
# NO-GC-NEXT:         InitExpr:
# NO-GC-NEXT:           Opcode:          I64_CONST
# NO-GC-NEXT:           Value:           123
# NO-GC-NEXT:       - Index:       2
# NO-GC-NEXT:         Type:        I64
# NO-GC-NEXT:         Mutable:     true
# NO-GC-NEXT:         InitExpr:
# NO-GC-NEXT:           Opcode:          I64_CONST
# NO-GC-NEXT:           Value:           456

# NO-GC:        - Type:            DATA
# NO-GC-NEXT:     Segments:
# NO-GC-NEXT:       - SectionOffset:   8
# NO-GC-NEXT:         InitFlags:       0
# NO-GC-NEXT:         Offset:
# NO-GC-NEXT:           Opcode:          I32_CONST
# NO-GC-NEXT:           Value:           65536
# NO-GC-NEXT:         Content:         '010000000000000002000000'
# NO-GC-NEXT:   - Type:            CUSTOM
# NO-GC-NEXT:     Name:            name
# NO-GC-NEXT:     FunctionNames:
# NO-GC-NEXT:       - Index:           0
# NO-GC-NEXT:         Name:            __wasm_call_ctors
# NO-GC-NEXT:       - Index:           1
# NO-GC-NEXT:         Name:            unused_function
# NO-GC-NEXT:       - Index:           2
# NO-GC-NEXT:         Name:            used_function
# NO-GC-NEXT:       - Index:           3
# NO-GC-NEXT:         Name:            _start
# NO-GC-NEXT:       - Index:           4
# NO-GC-NEXT:         Name:            use_global
# NO-GC-NEXT:     GlobalNames:
# NO-GC-NEXT:       - Index:           0
# NO-GC-NEXT:         Name:            __stack_pointer
# NO-GC-NEXT:       - Index:           1
# NO-GC-NEXT:         Name:            unused_global
# NO-GC-NEXT:       - Index:           2
# NO-GC-NEXT:         Name:            used_global
# NO-GC-NEXT:     DataSegmentNames:
# NO-GC-NEXT:       - Index:           0
# NO-GC-NEXT:         Name:            .data
# NO-GC-NEXT: ...

# RUN: not wasm-ld --gc-sections --relocatable -o %t1.no-gc.wasm %t.o 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
# CHECK-ERROR: error: -r and --gc-sections may not be used together
