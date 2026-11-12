# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 1.s -o 1.o
# RUN: ld.lld 1.o --shared -o 1.so
# RUN: llvm-readelf -d -s 1.so | FileCheck --check-prefix=CHECK1 %s

# CHECK1:      Symbol table '.dynsym'
# CHECK1:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def
# CHECK1:      Symbol table '.symtab'
# CHECK1:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64 2.s -o 2.o
# RUN: ld.lld 2.o --shared -o 2.so
# RUN: llvm-readelf -d -s 2.so | FileCheck --check-prefix=CHECK2 %s

# CHECK2:      0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# CHECK2:      Symbol table '.dynsym'
# CHECK2:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def
# CHECK2:      Symbol table '.symtab'
# CHECK2:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64 3.s -o 3.o
# RUN: ld.lld 3.o --shared -o 3.so
# RUN: llvm-readelf -d -s 3.so | FileCheck --check-prefix=CHECK3 %s

# CHECK3:      0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# CHECK3:      Symbol table '.dynsym'
# CHECK3:      0 IFUNC  GLOBAL DEFAULT [VARIANT_PCS] UND   ifunc_global_def
# CHECK3:      0 NOTYPE GLOBAL DEFAULT               [[#]] func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64 4.s -o 4.o
# RUN: ld.lld 4.o --shared -o 4.so
# RUN: llvm-readelf -d -s 4.so | FileCheck --check-prefix=CHECK4 %s

# CHECK4-NOT:  0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# CHECK4:      Symbol table '.dynsym'
# CHECK4:      0 IFUNC GLOBAL DEFAULT [VARIANT_PCS]  [[#]] ifunc_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64 5.s -o 5.o
# RUN: ld.lld 5.o --shared -o 5.so
# RUN: llvm-readelf -d -s 5.so | FileCheck --check-prefix=CHECK5 %s

# CHECK5:      Symbol table '.dynsym' contains 4 entries:
# CHECK5:      0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] UND   func_global_undef
# CHECK5-NEXT: 0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def
# CHECK5-NEXT: 0 IFUNC   GLOBAL DEFAULT [VARIANT_PCS] [[#]] ifunc_global_def
# CHECK5:      Symbol table '.symtab' contains 10 entries:
# CHECK5:      0 NOTYPE  LOCAL  DEFAULT [VARIANT_PCS] [[#]] func_local
# CHECK5-NEXT: 0 IFUNC   LOCAL  DEFAULT [VARIANT_PCS] [[#]] ifunc_local
# CHECK5:      0 NOTYPE  LOCAL  HIDDEN  [VARIANT_PCS] [[#]] func_global_hidden
# CHECK5-NEXT: 0 IFUNC   LOCAL  HIDDEN  [VARIANT_PCS] [[#]] ifunc_global_hidden
# CHECK5:      0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] [[#]] func_global_def
# CHECK5-NEXT: 0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] UND   func_global_undef
# CHECK5-NEXT: 0 IFUNC   GLOBAL DEFAULT [VARIANT_PCS] [[#]] ifunc_global_def

#--- 1.s
## An object with a variant_pcs symbol but without a R_AARCH64_JMP_SLOT
## should not generate a DT_AARCH64_VARIANT_PCS.
.text
.global func_global_def
.variant_pcs func_global_def

func_global_def:
  ret

#--- 2.s
## An object with a variant_pcs symbol and with a R_AARCH64_JMP_SLOT
## should generate a DT_AARCH64_VARIANT_PCS.
.text
.global func_global_def
.variant_pcs func_global_def

func_global_def:
  bl func_global_def

#--- 3.s
## Same as before, but targeting a GNU IFUNC.
.text
.global ifunc_global_def
.global func_global_def
.variant_pcs ifunc_global_def
.type ifunc_global_def, %gnu_indirect_function

func_global_def:
  bl ifunc_global_def

#--- 4.s
## An object with a variant_pcs symbol and with a R_AARCH64_IRELATIVE
## should not generate a DT_AARCH64_VARIANT_PCS.
.text
.global ifunc_global_def
.global func_global_def
.variant_pcs ifunc_global_def
.type ifunc_global_def, %gnu_indirect_function

ifunc_global_def:
  bl func_global_def

#--- 5.s
## Check if STO_AARCH64_VARIANT_PCS is kept on symbol st_other for both undef,
## local, and hidden visibility.
.text
.global func_global_def, func_global_undef, func_global_hidden
.global ifunc_global_def, ifunc_global_hidden
.local func_local

.hidden func_global_hidden, ifunc_global_hidden

.type ifunc_global_def, %gnu_indirect_function
.type ifunc_global_hidden, %gnu_indirect_function
.type ifunc_local, %gnu_indirect_function

.variant_pcs func_global_def
.variant_pcs func_global_undef
.variant_pcs func_global_hidden
.variant_pcs func_local
.variant_pcs ifunc_global_def
.variant_pcs ifunc_global_hidden
.variant_pcs ifunc_local

func_global_def:
func_global_hidden:
func_local:
ifunc_global_def:
ifunc_global_hidden:
ifunc_local:
  ret
