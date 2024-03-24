# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf %s -o %t.rv32_lp.o
# RUN: ld.lld %t.rv32_lp.o -zforce-zicfilp -o %t.rv32_lp | count 0
# RUN: llvm-readobj -n %t.rv32_lp | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP %s

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf %s -o %t.rv64_lp.o
# RUN: ld.lld %t.rv64_lp.o -zforce-zicfilp -o %t.rv64_lp | count 0
# RUN: llvm-readobj -n %t.rv64_lp | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP %s

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf %s -o %t.rv32_ss.o
# RUN: ld.lld %t.rv32_ss.o -zforce-zicfiss -o %t.rv32_ss | count 0
# RUN: llvm-readobj -n %t.rv32_ss | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFISS %s

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf %s -o %t.rv64_ss.o
# RUN: ld.lld %t.rv64_ss.o -zforce-zicfiss -o %t.rv64_ss | count 0
# RUN: llvm-readobj -n %t.rv64_ss | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFISS %s

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf %s -o %t.rv32_lp_ss.o
# RUN: ld.lld %t.rv32_lp_ss.o -zforce-zicfilp -zforce-zicfiss -o %t.rv32_lp_ss | count 0
# RUN: llvm-readobj -n %t.rv32_lp_ss | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP_ZICFISS %s

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf %s -o %t.rv64_lp_ss.o
# RUN: ld.lld %t.rv64_lp_ss.o -zforce-zicfilp -zforce-zicfiss -o %t.rv64_lp_ss | count 0
# RUN: llvm-readobj -n %t.rv64_lp_ss | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP_ZICFISS %s



// CHECK: Name: .note.gnu.property
// CHECK: Type: NT_GNU_PROPERTY_TYPE_0
// CHECK: Property [
// CHECK_ZICFISS: riscv feature: ZICFISS
// CHECK_ZICFILP: riscv feature: ZICFILP
// CHECK_ZICFILP_ZICFISS: riscv feature: ZICFILP, ZICFISS
// CHECK: ]
