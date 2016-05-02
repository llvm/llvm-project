# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o %S/Inputs/verneed1.so %S/Inputs/verneed2.so -o %t
# RUN: llvm-readobj -sections -dyn-symbols -dynamic-table %t | FileCheck %s
# RUN: llvm-objdump -s %t | FileCheck --check-prefix=CONTENTS %s

# CHECK:        Index: 2
# CHECK-NEXT:   Name: .gnu.version (9)
# CHECK-NEXT:   Type: SHT_GNU_versym (0x6FFFFFFF)
# CHECK-NEXT:   Flags [ (0x2)
# CHECK-NEXT:     SHF_ALLOC (0x2)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x10228
# CHECK-NEXT:   Offset: 0x228
# CHECK-NEXT:   Size: 8
# CHECK-NEXT:   Link: 0
# CHECK-NEXT:   Info: 0
# CHECK-NEXT:   AddressAlignment: 2
# CHECK-NEXT:   EntrySize: 2
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index: 3
# CHECK-NEXT:   Name: .gnu.version_r (22)
# CHECK-NEXT:   Type: SHT_GNU_verneed (0x6FFFFFFE)
# CHECK-NEXT:   Flags [ (0x2)
# CHECK-NEXT:     SHF_ALLOC (0x2)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x10230
# CHECK-NEXT:   Offset: 0x230
# CHECK-NEXT:   Size: 80
# CHECK-NEXT:   Link: 5
# CHECK-NEXT:   Info: 2
# CHECK-NEXT:   AddressAlignment: 4
# CHECK-NEXT:   EntrySize: 0
# CHECK-NEXT: }

# CHECK:      DynamicSymbols [
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: @ (0)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local (0x0)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: f1@v3 (1)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global (0x1)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: f2@v2 (21)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global (0x1)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: g1@v1 (27)
# CHECK-NEXT:     Value: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Global (0x1)
# CHECK-NEXT:     Type: None (0x0)
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: Undefined (0x0)
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK:      0x000000006FFFFFF0 VERSYM               0x10228
# CHECK-NEXT: 0x000000006FFFFFFE VERNEED              0x10230
# CHECK-NEXT: 0x000000006FFFFFFF VERNEEDNUM           2

# CONTENTS:      Contents of section .gnu.version:
# CONTENTS-NEXT:  10228 00000200 03000400
# CONTENTS-NEXT: Contents of section .gnu.version_r:
#                       vn_version
#                           vn_cnt
#                                vn_file  vn_aux   vn_next
# CONTENTS-NEXT:  10230 01000200 04000000 20000000 10000000  ........ .......
# CONTENTS-NEXT:  10240 01000100 1e000000 30000000 00000000  ........0.......
#                       vna_hash vna_flags
#                                    vna_other
#                                         vna_name
#                                                  vna_next
# CONTENTS-NEXT:  10250 92070000 00000300 18000000 10000000  ................
# CONTENTS-NEXT:  10260 93070000 00000200 12000000 00000000  ................
# CONTENTS-NEXT:  10270 91070000 00000400 2c000000 00000000  ........,.......
# CONTENTS:      Contents of section .dynstr:
# CONTENTS-NEXT:  102a8 00663100 7665726e 65656431 2e736f2e  .f1.verneed1.so.
# CONTENTS-NEXT:  102b8 30007633 00663200 76320067 31007665  0.v3.f2.v2.g1.ve
# CONTENTS-NEXT:  102c8 726e6565 64322e73 6f2e3000 763100    rneed2.so.0.v1.

.globl _start
_start:
call f1@plt
call f2@plt
call g1@plt
