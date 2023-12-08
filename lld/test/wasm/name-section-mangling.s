# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld --export=_Z3fooi --demangle -o %t_demangle.wasm %t.o
# RUN: obj2yaml %t_demangle.wasm | FileCheck --check-prefixes=CHECK,DEMANGLE %s
# RUN: wasm-ld --export=_Z3fooi --no-demangle -o %t_nodemangle.wasm %t.o
# RUN: obj2yaml %t_nodemangle.wasm | FileCheck --check-prefixes=CHECK,MANGLE %s

# Check that the EXPORT name is still mangled, but that the "name" custom
# section contains the unmangled name.

.globl _start
.globl _Z3fooi
.weak _Z3bari

.functype _Z3bari (i32) -> ()

_Z3fooi:
  .functype _Z3fooi (i32) -> ()
  end_function

_start:
  .functype _start () -> ()
  i32.const 1
  call _Z3fooi
  i32.const 1
  call _Z3bari
  end_function

# CHECK:        - Type:            EXPORT
# CHECK-NEXT:     Exports:
# CHECK-NEXT:       - Name:            memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:       - Name:            _start
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:       - Name:            _Z3fooi
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Index:           2
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            000B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            410110828080800041011080808080000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            0B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            name
# CHECK-NEXT:     FunctionNames:
# CHECK-NEXT:       - Index:           0
# DEMANGLE-NEXT:      Name:            'undefined_weak:bar(int)'
# MANGLE-NEXT:        Name:            'undefined_weak:_Z3bari'
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            _start
# CHECK-NEXT:       - Index:           2
# DEMANGLE-NEXT:      Name:            'foo(int)'
# MANGLE-NEXT:        Name:            _Z3fooi
# CHECK-NEXT:     GlobalNames:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            __stack_pointer
# CHECK-NEXT: ...
