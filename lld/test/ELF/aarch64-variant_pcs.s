# REQUIRES: aarch64
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %t/test1 -o %t.o
# RUN: ld.lld %t.o --shared -o %t.so
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix T1-PCSDYN %s
# RUN: llvm-readelf --symbols %t.so | FileCheck --check-prefix T1-PCSSYM %s

# T1-PCSDYN-NOT:  0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# T1-PCSSYM:      Symbol table '.dynsym'
# T1-PCSSYM:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %t/test2 -o %t.o
# RUN: ld.lld %t.o --shared -o %t.so
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix T2-PCSDYN %s
# RUN: llvm-readelf --symbols %t.so | FileCheck --check-prefix T2-PCSSYM %s

# T2-PCSDYN:      0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# T2-PCSSYM:      Symbol table '.dynsym'
# T2-PCSSYM:      0 NOTYPE GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %t/test3 -o %t.o
# RUN: ld.lld %t.o --shared -o %t.so
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix T3-PCSDYN %s
# RUN: llvm-readelf --symbols %t.so | FileCheck --check-prefix T3-PCSSYM %s

# T3-PCSDYN:      0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# T3-PCSSYM:      Symbol table '.dynsym'
# T3-PCSSYM:      0 IFUNC  GLOBAL DEFAULT [VARIANT_PCS] UND   pcs_ifunc_global_def
# T3-PCSSYM:      0 NOTYPE GLOBAL DEFAULT               [[#]] pcs_func_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %t/test4 -o %t.o
# RUN: ld.lld %t.o --shared -o %t.so
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix T4-PCSDYN %s
# RUN: llvm-readelf --symbols %t.so | FileCheck --check-prefix T4-PCSSYM %s

# T4-PCSDYN-NOT:  0x0000000070000005 (AARCH64_VARIANT_PCS) 0
# T4-PCSSYM:      Symbol table '.dynsym'
# T4-PCSSYM:      0 IFUNC GLOBAL DEFAULT [VARIANT_PCS]  [[#]] pcs_ifunc_global_def

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %t/test5 -o %t.o
# RUN: ld.lld %t.o --shared -o %t.so
# RUN: llvm-readelf --symbols %t.so | FileCheck --check-prefix T5-PCSSYM %s

# T5-PCSSYM:      Symbol table '.dynsym'
# T5-PCSSYM:      0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] UND   pcs_func_global_undef
# T5-PCSSYM-NEXT: 0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_func_global_def
# T5-PCSSYM-NEXT: 0 IFUNC   GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_ifunc_global_def
# T5-PCSSYM:      Symbol table '.symtab' contains 10 entries:
# T5-PCSSYM:      0 NOTYPE  LOCAL  DEFAULT [VARIANT_PCS] [[#]] pcs_func_local
# T5-PCSSYM-NEXT: 0 IFUNC   LOCAL  DEFAULT [VARIANT_PCS] [[#]] pcs_ifunc_local
# T5-PCSSYM:      0 NOTYPE  LOCAL  HIDDEN  [VARIANT_PCS] [[#]] pcs_func_global_hidden
# T5-PCSSYM-NEXT: 0 IFUNC   LOCAL  HIDDEN  [VARIANT_PCS] [[#]] pcs_ifunc_global_hidden
# T5-PCSSYM:      0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_func_global_def
# T5-PCSSYM-NEXT: 0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] UND   pcs_func_global_undef
# T5-PCSSYM-NEXT: 0 IFUNC   GLOBAL DEFAULT [VARIANT_PCS] [[#]] pcs_ifunc_global_def


#--- test1
## An object with a variant_pcs symbol but without a R_AARCH64_JMP_SLOT
## should not generate a DT_AARCH64_VARIANT_PCS.
.text
.global pcs_func_global_def
.variant_pcs pcs_func_global_def

pcs_func_global_def:
  ret

#--- test2
## An object with a variant_pcs symbol and with a R_AARCH64_JMP_SLOT
## should generate a DT_AARCH64_VARIANT_PCS.
.text
.global pcs_func_global_def
.variant_pcs pcs_func_global_def

pcs_func_global_def:
  bl pcs_func_global_def

#--- test3
## Same as before, but targeting a GNU IFUNC.
.text
.global pcs_ifunc_global_def
.global pcs_func_global_def
.variant_pcs pcs_ifunc_global_def
.type pcs_ifunc_global_def, %gnu_indirect_function

pcs_func_global_def:
  bl pcs_ifunc_global_def

#--- test4
## An object with a variant_pcs symbol and with a R_AARCH64_IRELATIVE
## should not generate a DT_AARCH64_VARIANT_PCS.
.text
.global pcs_ifunc_global_def
.global pcs_func_global_def
.variant_pcs pcs_ifunc_global_def
.type pcs_ifunc_global_def, %gnu_indirect_function

pcs_ifunc_global_def:
  bl pcs_func_global_def

#--- test5
## Check if STO_AARCH64_VARIANT_PCS is kept on symbol st_other for both undef,
## local, and hidden visibility.
.text
.global pcs_func_global_def, pcs_func_global_undef, pcs_func_global_hidden
.global pcs_ifunc_global_def, pcs_ifunc_global_hidden
.local pcs_func_local

.hidden pcs_func_global_hidden, pcs_ifunc_global_hidden

.type pcs_ifunc_global_def, %gnu_indirect_function
.type pcs_ifunc_global_hidden, %gnu_indirect_function
.type pcs_ifunc_local, %gnu_indirect_function

.variant_pcs pcs_func_global_def
.variant_pcs pcs_func_global_undef
.variant_pcs pcs_func_global_hidden
.variant_pcs pcs_func_local
.variant_pcs pcs_ifunc_global_def
.variant_pcs pcs_ifunc_global_hidden
.variant_pcs pcs_ifunc_local

pcs_func_global_def:
pcs_func_global_hidden:
pcs_func_local:
pcs_ifunc_global_def:
pcs_ifunc_global_hidden:
pcs_ifunc_local:
  ret
