# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --no-check-features -o %t1.wasm %t.o
# RUN: obj2yaml %t1.wasm | FileCheck %s

.global g1
.import_module g1, env
.import_name g1, g
.globaltype g1, i64, immutable

# Same import module/name/type as `g1`, should be de-duped.
.global g2
.import_module g2, env
.import_name g2, g
.globaltype g2, i64, immutable

# Imported as an i32 instead of i64, so should not be de-duped.
.global g3
.import_module g3, env
.import_name g3, g
.globaltype g3, i32, immutable

# Imported as mutable instead of immutable, so should not be de-duped.
.global g4
.import_module g4, env
.import_name g4, g
.globaltype g4, i64

.globl _start
_start:
  .functype _start () -> ()
  global.get g1
  drop
  global.get g2
  drop
  global.get g3
  drop
  global.get g4
  drop
  end_function


# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           g
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I64
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           g
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I32
# CHECK-NEXT:        GlobalMutable:   false
# CHECK-NEXT:      - Module:          env
# CHECK-NEXT:        Field:           g
# CHECK-NEXT:        Kind:            GLOBAL
# CHECK-NEXT:        GlobalType:      I64
# CHECK-NEXT:        GlobalMutable:   true
# CHECK-NEXT:  - Type:

# CHECK:         GlobalNames:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Name:            g1
# CHECK-NEXT:      - Index:           1
# CHECK-NEXT:        Name:            g3
# CHECK-NEXT:      - Index:           2
# CHECK-NEXT:        Name:            g4
# CHECK-NEXT:      - Index:           3
# CHECK-NEXT:        Name:            __stack_pointer
