# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

# RUN: wasm-ld -o %t.no-export.wasm %t.o
# RUN: obj2yaml %t.no-export.wasm | FileCheck %s --check-prefix=NO-EXPORT

# RUN: wasm-ld --export-all -o %t.export.wasm %t.o
# RUN: obj2yaml %t.export.wasm | FileCheck %s --check-prefix=EXPORT

# RUN: wasm-ld --export-all --no-gc-sections -o %t.export-no-gc.wasm %t.o
# RUN: obj2yaml %t.export-no-gc.wasm | FileCheck %s --check-prefix=EXPORT

.globaltype __stack_pointer, i32

# Hidden function, referenced
.globl foo
.hidden foo
foo:
  .functype foo () -> ()
  end_function

# Hidden function, unreferenced
.globl bar
.hidden bar
bar:
  .functype bar () -> ()
  end_function

# Local function, referenced
local_func:
  .functype local_func () -> ()
  end_function

# Local function, unreferenced
local_unref:
  .functype local_unref () -> ()
  end_function

.globl _start
_start:
  .functype _start () -> ()
  i32.const 3
  global.set __stack_pointer
  call foo
  call local_func
  end_function

# NO-EXPORT:      - Type:            EXPORT
# NO-EXPORT-NEXT:   Exports:
# NO-EXPORT-NEXT:     - Name:            memory
# NO-EXPORT-NEXT:       Kind:            MEMORY
# NO-EXPORT-NEXT:       Index:           0
# NO-EXPORT-NEXT:     - Name:            _start
# NO-EXPORT-NEXT:       Kind:            FUNCTION
# NO-EXPORT-NEXT:       Index:           2
# NO-EXPORT-NOT:        Name:            foo
# NO-EXPORT-NOT:        Name:            bar
# NO-EXPORT-NOT:        Name:            local_func
# NO-EXPORT:      - Type:            CODE

# EXPORT:         - Type:            EXPORT
# EXPORT-NEXT:      Exports:
# EXPORT-NEXT:        - Name:            memory
# EXPORT-NEXT:          Kind:            MEMORY
# EXPORT-NEXT:          Index:           0
# EXPORT-NEXT:        - Name:            __wasm_call_ctors
# EXPORT-NEXT:          Kind:            FUNCTION
# EXPORT-NEXT:          Index:           {{[0-9]+}}
# EXPORT-NEXT:        - Name:            foo
# EXPORT-NEXT:          Kind:            FUNCTION
# EXPORT-NEXT:          Index:           {{[0-9]+}}
# EXPORT-NEXT:        - Name:            bar
# EXPORT-NEXT:          Kind:            FUNCTION
# EXPORT-NEXT:          Index:           {{[0-9]+}}
# EXPORT-NEXT:        - Name:            _start
# EXPORT-NEXT:          Kind:            FUNCTION
# EXPORT-NEXT:          Index:           {{[0-9]+}}
# EXPORT-NEXT:        - Name:            __dso_handle
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           4
# EXPORT-NEXT:        - Name:            __data_end
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           5
# EXPORT-NEXT:        - Name:            __rodata_start
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           6
# EXPORT-NEXT:        - Name:            __rodata_end
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           7
# EXPORT-NEXT:        - Name:            __stack_low
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           8
# EXPORT-NEXT:        - Name:            __stack_high
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           9
# EXPORT-NEXT:        - Name:            __global_base
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           10
# EXPORT-NEXT:        - Name:            __heap_base
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           11
# EXPORT-NEXT:        - Name:            __heap_end
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           12
# EXPORT-NEXT:        - Name:            __memory_base
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           1
# EXPORT-NEXT:        - Name:            __table_base
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           2
# EXPORT-NEXT:        - Name:            __wasm_first_page_end
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           13
# EXPORT-NEXT:        - Name:            __tls_base
# EXPORT-NEXT:          Kind:            GLOBAL
# EXPORT-NEXT:          Index:           3
# EXPORT-NOT:         Name:            local_func
# EXPORT-NOT:         Name:            local_unref
# EXPORT:         - Type:            CODE
