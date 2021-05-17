# REQUIRES: aarch64
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %t/test.s -o %t.o
# RUN: ld.lld -shared --script %t/script %t.o -o %t.so
# RUN: ld.lld -pie --script %t/script %t.o -o %t.exe
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOCS-SHARED %s
# RUN: llvm-readobj -r %t.exe | FileCheck --check-prefix=RELOCS-PIE %s
# RUN: llvm-objdump --no-show-raw-insn -d %t.so | FileCheck --check-prefix=DISAS %s

## Check if the R_AARCH64_LD64_GOTPAGE_LO15 generates the GOT entries.
# RELOCS-SHARED:      Relocations [
# RELOCS-SHARED-NEXT:   Section (5) .rela.dyn {
# RELOCS-SHARED-NEXT:     0x{{[0-9A-F]+}} R_AARCH64_GLOB_DAT global1 0x{{[0-9A-F]+}}
# RELOCS-SHARED-NEXT:     0x{{[0-9A-F]+}} R_AARCH64_GLOB_DAT global2 0x{{[0-9A-F]+}}
# RELOCS-SHARED-NEXT:   }
# RELOCS-SHARED-NEXT: ]

# RELOCS-PIE:      Relocations [
# RELOCS-PIE-NEXT:   Section (5) .rela.dyn {
# RELOCS-PIE-NEXT:     0x{{[0-9A-F]+}} R_AARCH64_RELATIVE - 0x{{[0-9A-F]+}}
# RELOCS-PIE-NEXT:     0x{{[0-9A-F]+}} R_AARCH64_RELATIVE - 0x{{[0-9A-F]+}}
# RELOCS-PIE-NEXT:   }
# RELOCS-PIE-NEXT: ]

# DISAS:      adrp    x0, 0xf000
# DISAS-NEXT: ldr     x0, [x0, #4088]
# DISAS-NEXT: ldr     x1, [x0, #4096]

#--- script
SECTIONS {
  .got (0x10000 - 8) : { *.got }
}

#--- test.s
.globl	_start
.type	_start,@function
_start:
	adrp    x0, _GLOBAL_OFFSET_TABLE_
	ldr     x0, [x0, #:gotpage_lo15:global1]
	ldr     x1, [x0, #:gotpage_lo15:global2]

.type   global1,@object
.comm   global1,8,8
.type   global2,@object
.comm   global2,8,8
