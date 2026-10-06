# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/define-tls.o %t/define-tls.s
# RUN: wasm-ld -shared -o %t/define-tls.so %t/define-tls.o
# RUN: obj2yaml %t/define-tls.so | FileCheck %s

#--- define-tls.s

.section  .tdata.tls1,"",@
.globl  tls1
.p2align  2
tls1:
  .int32  1
  .size tls1, 4

.section  .custom_section.target_features,"",@
  .int8 3
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 15
  .ascii "mutable-globals"

#--- use-tls.s

  .globl  _start
_start:
  .functype _start () -> ()
  i32.const tls1@TLSREL
  drop
  end_function

.section  .custom_section.target_features,"",@
  .int8 3
  .int8 43
  .int8 7
  .ascii  "atomics"
  .int8 43
  .int8 11
  .ascii  "bulk-memory"
  .int8 43
  .int8 15
  .ascii "mutable-globals"


#      CHECK:    Name:            dylink.0
# CHECK-NEXT:    TargetArch:      wasm32
#      CHECK:    ExportInfo:
# CHECK-NEXT:      - Name:            tls1
# CHECK-NEXT:        Flags:           [ TLS ]
# CHECK-NEXT:  - Type:            TYPE

#      CHECK:  - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Type:            I32
# CHECK-NEXT:        Mutable:         false
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           0

#      CHECK:  - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            __wasm_call_ctors
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            tls1
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        Index:           2

# Check that linking a shared object that exports a TLS global doesn't cause
# the global to be erroneously marked as non-TLS. The reference itself is
# still rejected, since TLS relocations resolve against the referencing
# module's own __tls_base and the dynamic linking ABI cannot express a
# reference into another module's TLS block.

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/use-tls.o %t/use-tls.s
# RUN: not wasm-ld --shared-memory -pie -o %t/use-tls.wasm %t/define-tls.so %t/use-tls.o 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: {{.*}}use-tls.o: relocation R_WASM_MEMORY_ADDR_TLS_SLEB cannot be used against symbol `tls1` defined in shared library {{.*}}define-tls.so; thread-local variables cannot be accessed across modules
