# REQUIRES: x86
## TYPE=<value> customizes the output section type.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/mismatch.s -o %t/mismatch.o
# RUN: ld.lld -T %t/a.lds %t/a.o -o %t/a
# RUN: llvm-readelf -S %t/a | FileCheck %s

# RUN: ld.lld -r -T %t/a.lds %t/a.o -o %t/a.ro
# RUN: llvm-readelf -S %t/a.ro | FileCheck %s

# CHECK:       [Nr] Name              Type            Address          Off      Size   ES Flg Lk Inf Al
# CHECK-NEXT:  [ 0]                   NULL            [[#%x,]]         [[#%x,]] 000000 00      0   0  0
# CHECK-NEXT:  [ 1] progbits          PROGBITS        [[#%x,]]         [[#%x,]] 000001 00   A  0   0  1
# CHECK-NEXT:  [ 2] note              NOTE            [[#%x,]]         [[#%x,]] 000002 00   A  0   0  1
# CHECK-NEXT:  [ 3] nobits            NOBITS          [[#%x,]]         [[#%x,]] 000001 00   A  0   0  1
# CHECK-NEXT:  [ 4] init_array        INIT_ARRAY      [[#%x,]]         [[#%x,]] 000008 00   A  0   0  1
# CHECK-NEXT:  [ 5] fini_array        FINI_ARRAY      [[#%x,]]         [[#%x,]] 000008 00   A  0   0  1
# CHECK-NEXT:  [ 6] preinit_array     PREINIT_ARRAY   [[#%x,]]         [[#%x,]] 000008 00   A  0   0  1
# CHECK-NEXT:  [ 7] group             GROUP           [[#%x,]]         [[#%x,]] 000004 00   A [[#SYMTAB:]] 0  1
# CHECK-NEXT:  [ 8] expr              0x42: <unknown> [[#%x,]]         [[#%x,]] 000001 00   A  0   0  1
# CHECK:       [[[#SYMTAB]]] .symtab  SYMTAB

# RUN: not ld.lld -T %t/a.lds %t/a.o %t/mismatch.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR1

# ERR1:      error: section type mismatch for progbits
# ERR1-NEXT: >>> {{.*}}.o:(progbits): SHT_NOTE
# ERR1-NEXT: >>> output section progbits: SHT_PROGBITS
# ERR1:      error: section type mismatch for expr
# ERR1-NEXT: >>> {{.*}}.o:(expr): Unknown
# ERR1-NEXT: >>> output section expr: Unknown

# RUN: ld.lld -T %t/a.lds %t/a.o %t/mismatch.o -o %t/mismatch --noinhibit-exec
# RUN: llvm-readelf -S %t/mismatch | FileCheck %s --check-prefix=MISMATCH

## Mismatched progbits and expr are changed to SHT_PROGBITS.
# MISMATCH: progbits PROGBITS
# MISMATCH: note     NOTE
# MISMATCH: expr     0x42: <unknown>

# RUN: not ld.lld -T %t/unknown1.lds %t/a.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN1
# UNKNOWN1: error: {{.*}}.lds:1: symbol not found: foo

## For a symbol named SHT_*, give a better diagnostic.
# RUN: not ld.lld -T %t/unknown2.lds %t/a.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN2
# UNKNOWN2: error: {{.*}}.lds:1: unknown section type SHT_DYNAMIC

# RUN: not ld.lld -T %t/parseerr1.lds %t/a.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=PARSEERR1
# PARSEERR1: error: {{.*}}.lds:1: = expected, but got )

#--- a.s
.globl _start, myinit
_start:
  ret
myinit:
  ret

## Compatible with TYPE = SHT_NOTE below.
.section note,"a",@note
.byte 0

#--- a.lds
SECTIONS {
  progbits (TYPE=SHT_PROGBITS) : { BYTE(1) }
  note (TYPE = SHT_NOTE) : { BYTE(7) *(note) }
  nobits ( TYPE=SHT_NOBITS) : { BYTE(8) }
  init_array (TYPE=SHT_INIT_ARRAY ) : { QUAD(myinit) }
  fini_array (TYPE=SHT_FINI_ARRAY) : { QUAD(15) }
  preinit_array . (TYPE=SHT_PREINIT_ARRAY) : { QUAD(16) }
  group (TYPE=17) : { LONG(17) }
  expr (TYPE=0x41+1) : { BYTE(0x42) *(expr) }
}

#--- mismatch.s
.section progbits,"a",@note
.byte 0

.section expr,"a",@0x60000000
.byte 0

#--- unknown1.lds
SECTIONS { err (TYPE=foo) : {} }

#--- unknown2.lds
SECTIONS { err (TYPE=SHT_DYNAMIC) : {} }

#--- parseerr1.lds
SECTIONS { err (TYPE) : {} }
