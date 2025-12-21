# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/ret32.s -o %t.ret32.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/Inputs/call-ret32.s -o %t.call.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.main.o

# RUN: wasm-ld --export=call_ret32 --export=ret32 -o %t.wasm %t.main.o %t.ret32.o %t.call.o 2>&1 | FileCheck %s -check-prefix=WARN
# RUN: obj2yaml %t.wasm | FileCheck %s -check-prefix=YAML

# RUN: wasm-ld -r -o %t.reloc.o %t.main.o %t.ret32.o %t.call.o 2>&1 | FileCheck %s -check-prefix=WARN
# RUN: obj2yaml %t.reloc.o | FileCheck %s -check-prefix=RELOC

# RUN: rm -f %t.a
# RUN: llvm-ar crS %t.a %t.ret32.o %t.call.o
# RUN: wasm-ld --export=call_ret32 --export=ret32 -o %t2.wasm %t.main.o %t.a 2>&1 | FileCheck %s -check-prefix=ARCHIVE
# RUN: obj2yaml %t2.wasm | FileCheck %s -check-prefix=YAML

# RUN: not wasm-ld --fatal-warnings -o %t.wasm %t.main.o %t.ret32.o %t.call.o 2>&1 | FileCheck %s -check-prefix=ERROR

.functype ret32 (i32, i64, i32) -> (i32)

.hidden  _start
.globl  _start
_start:
  .functype _start () -> ()
  i32.const 1
  i64.const 2
  i32.const 3
  call  ret32
  drop
  i32.const 1
  i64.const 2
  i32.const 3
  i32.const 0
  i32.load  ret32_address_main
  call_indirect  (i32, i64, i32) -> (i32)
  drop
  end_function

.section  .data,"",@
.globl  ret32_address_main
.p2align  2

ret32_address_main:
  .int32  ret32
.size ret32_address_main, 4

# WARN: warning: function signature mismatch: ret32
# WARN-NEXT: >>> defined as (i32, i64, i32) -> i32 in {{.*}}.main.o
# WARN-NEXT: >>> defined as (f32) -> i32 in {{.*}}.ret32.o

# ARCHIVE: warning: function signature mismatch: ret32
# ARCHIVE-NEXT: >>> defined as (i32, i64, i32) -> i32 in {{.*}}.main.o
# ARCHIVE-NEXT: >>> defined as (f32) -> i32 in {{.*}}.ret32.o

# ERROR: error: function signature mismatch: ret32
# ERROR-NEXT: >>> defined as (i32, i64, i32) -> i32 in {{.*}}.main.o
# ERROR-NEXT: >>> defined as (f32) -> i32 in {{.*}}.ret32.o

# YAML:        - Type:            EXPORT
# YAML:           - Name:            ret32
# YAML-NEXT:        Kind:            FUNCTION
# YAML-NEXT:        Index:           2
# YAML-NEXT:      - Name:            call_ret32
# YAML-NEXT:        Kind:            FUNCTION
# YAML-NEXT:        Index:           3

# YAML:        - Type:            CUSTOM
# YAML-NEXT:     Name:            name
# YAML-NEXT:     FunctionNames:
# YAML-NEXT:       - Index:           0
# YAML-NEXT:         Name:            'signature_mismatch:ret32'
# YAML-NEXT:       - Index:           1
# YAML-NEXT:         Name:            _start
# YAML-NEXT:       - Index:           2
# YAML-NEXT:         Name:            ret32
# YAML-NEXT:       - Index:           3
# YAML-NEXT:         Name:            call_ret32
# YAML-NEXT:     GlobalNames:
# YAML-NEXT:       - Index:           0
# YAML-NEXT:         Name:            __stack_pointer
# YAML-NEXT:     DataSegmentNames:
# YAML-NEXT:       - Index:           0
# YAML-NEXT:         Name:            .data
# YAML-NEXT: ...

#      RELOC:     Name:            linking
# RELOC-NEXT:     Version:         2
# RELOC-NEXT:     SymbolTable:
# RELOC-NEXT:       - Index:           0
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            _start
# RELOC-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# RELOC-NEXT:         Function:        1
# RELOC-NEXT:       - Index:           1
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            ret32
# RELOC-NEXT:         Flags:           [ ]
# RELOC-NEXT:         Function:        2
# RELOC-NEXT:       - Index:           2
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            ret32_address_main
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Segment:         0
# RELOC-NEXT:         Size:            4
# RELOC-NEXT:       - Index:           3
# RELOC-NEXT:         Kind:            TABLE
# RELOC-NEXT:         Name:            __indirect_function_table
# RELOC-NEXT:         Flags:           [ UNDEFINED, NO_STRIP ]
# RELOC-NEXT:         Table:           0
# RELOC-NEXT:       - Index:           4
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            call_ret32
# RELOC-NEXT:         Flags:           [ ]
# RELOC-NEXT:         Function:        3
# RELOC-NEXT:       - Index:           5
# RELOC-NEXT:         Kind:            DATA
# RELOC-NEXT:         Name:            ret32_address
# RELOC-NEXT:         Flags:           [  ]
# RELOC-NEXT:         Segment:         1
# RELOC-NEXT:         Size:            4
# RELOC-NEXT:       - Index:           6
# RELOC-NEXT:         Kind:            FUNCTION
# RELOC-NEXT:         Name:            'signature_mismatch:ret32'
# RELOC-NEXT:         Flags:           [ BINDING_LOCAL ]
# RELOC-NEXT:         Function:        0
