# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %p/Inputs/dwarf5-main-addr-section-reuse.s    -o %tmain.o
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %p/Inputs/dwarf5-helper1-addr-section-reuse.s -o %thelper1.o
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %p/Inputs/dwarf5-helper2-addr-section-reuse.s -o %thelper2.o
# RUN: %clang %cflags -dwarf-5 %tmain.o %thelper1.o %thelper2.o -o %t.exe -Wl,-q
# RUN: llvm-dwarfdump --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-bolt %t.exe -o %t.exe.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-info %t.exe.bolt | FileCheck --check-prefix=POSTCHECK %s

## This test checks that when a binary is bolted if CU is not modified and has DW_AT_addr_base that is shared
## after being bolted CUs still share same entry in .debug_addr.

# PRECHECK: DW_AT_addr_base (0x00000008)
# PRECHECK: DW_AT_addr_base (0x00000008)
# PRECHECK: DW_AT_addr_base (0x00000008)

# POSTCHECK: DW_AT_addr_base (0x00000008)
# POSTCHECK: DW_AT_addr_base (0x00000020)
# POSTCHECK: DW_AT_addr_base (0x00000020)
