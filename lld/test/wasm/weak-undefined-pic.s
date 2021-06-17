# Checks handling of undefined weak external functions.  When the
# static linker decides they are undefined, check GOT relocations
# resolve to zero (i.e. a global that contains zero.).
#
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o -o %t1.wasm
# RUN: obj2yaml %t1.wasm | FileCheck %s
#
# With `--unresolved-symbols=ignore-all` the behaviour should be the same
# as the default.>
#
# RUN: wasm-ld --unresolved-symbols=ignore-all %t.o -o %t2.wasm
# RUN: obj2yaml %t2.wasm | FileCheck %s

.globl get_foo_addr
get_foo_addr:
  .functype get_foo_addr () -> (i32)
  global.get foo@GOT
  end_function

.globl _start
_start:
  .functype _start () -> ()
  call get_foo_addr
  call foo
  end_function

.weak foo
.functype foo () -> (i32)

# Verify that we do not generate dynamic relocations for the GOT entry.

# CHECK-NOT: __wasm_apply_global_relocs

# Verify that we do not generate an import for foo

# CHECK-NOT:  - Type:            IMPORT

#      CHECK:   - Type:            GLOBAL
# CHECK-NEXT:     Globals:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         true
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           66560
# Global 'undefined_weak:foo' representing the GOT entry for foo
# Unlike other internal GOT entries that need to be mutable this one
# is immutable and not updated by `__wasm_apply_global_relocs`
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Type:            I32
# CHECK-NEXT:         Mutable:         false
# CHECK-NEXT:         InitExpr:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0

#      CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            name
# CHECK-NEXT:    FunctionNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            'undefined_weak:foo'
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            get_foo_addr
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            _start
# CHECK-NEXT:    GlobalNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            __stack_pointer
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            'GOT.func.internal.undefined_weak:foo'

# With `-pie` or `-shared` the resolution should be deferred to the dynamic
# linker and the function address should be imported as GOT.func.foo.
#
# RUN: wasm-ld --experimental-pic -pie %t.o -o %t3.wasm
# RUN: obj2yaml %t3.wasm | FileCheck %s --check-prefix=IMPORT

#      IMPORT:  - Type:            IMPORT
#      IMPORT:        Field:           foo
# IMPORT-NEXT:        Kind:            FUNCTION
# IMPORT-NEXT:        SigIndex:        0
# IMPORT-NEXT:      - Module:          GOT.func
# IMPORT-NEXT:        Field:           foo
# IMPORT-NEXT:        Kind:            GLOBAL
# IMPORT-NEXT:        GlobalType:      I32
# IMPORT-NEXT:        GlobalMutable:   true

#      IMPORT:     GlobalNames:
# IMPORT-NEXT:       - Index:           0
# IMPORT-NEXT:         Name:            __memory_base
# IMPORT-NEXT:       - Index:           1
# IMPORT-NEXT:         Name:            __table_base
# IMPORT-NEXT:       - Index:           2
# IMPORT-NEXT:         Name:            foo
