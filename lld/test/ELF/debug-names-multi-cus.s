# REQUIRES: x86
## Test name indexes that contain multiple CU offsets due to LTO.

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-bcd.s -o bcd.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-ef.s -o ef.o
# RUN: ld.lld --debug-names a.o bcd.o ef.o -o out
# RUN: llvm-dwarfdump --debug-info --debug-names out | FileCheck %s --check-prefix=DWARF

## Place the multiple CU offsets in the second name index in an input file.
# RUN: ld.lld -r a.o bcd.o -o abcd.o
# RUN: ld.lld --debug-names abcd.o ef.o -o out
# RUN: llvm-dwarfdump --debug-info --debug-names out | FileCheck %s --check-prefix=DWARF

# DWARF:      0x00000000: Compile Unit
# DWARF:      0x00000040: Compile Unit
# DWARF:      0x0000006e: Compile Unit
# DWARF:      0x00000098: Compile Unit
# DWARF:      0x000000c6: Compile Unit
# DWARF:      0x000000f4: Compile Unit

# DWARF:      Compilation Unit offsets [
# DWARF-NEXT:   CU[0]: 0x00000000
# DWARF-NEXT:   CU[1]: 0x00000040
# DWARF-NEXT:   CU[2]: 0x0000006e
# DWARF-NEXT:   CU[3]: 0x00000098
# DWARF-NEXT:   CU[4]: 0x000000c6
# DWARF-NEXT:   CU[5]: 0x000000f4
# DWARF-NEXT: ]
# DWARF:        String: {{.*}} "vc"
# DWARF:          DW_IDX_compile_unit: 0x02
# DWARF:        String: {{.*}} "vd"
# DWARF:          DW_IDX_die_offset:
# DWARF-SAME:                        0x00000020
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x03
# DWARF:        String: {{.*}} "ve"
# DWARF:          DW_IDX_die_offset:
# DWARF-SAME:                        0x0000001e
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x04
# DWARF:        String: {{.*}} "vf"
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x05
# DWARF:        String: {{.*}} "vb"
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x01
