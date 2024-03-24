// When -march with zicfiss0p4 or zicfilp0p4 add GNU property to file object

// RUN: %clang --target=riscv32-linux-gnu -menable-experimental-extensions -march=rv32gc_zicfiss0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFISS %s
// RUN: %clang --target=riscv64-linux-gnu -menable-experimental-extensions -march=rv64gc_zicfiss0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFISS %s
// RUN: %clang --target=riscv32-linux-gnu -menable-experimental-extensions -march=rv32gc_zicfilp0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP %s
// RUN: %clang --target=riscv64-linux-gnu -menable-experimental-extensions -march=rv64gc_zicfilp0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP %s
// RUN: %clang --target=riscv32-linux-gnu -menable-experimental-extensions -march=rv32gc_zicfilp0p4_zicfiss0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP_ZICFISS %s
// RUN: %clang --target=riscv64-linux-gnu -menable-experimental-extensions -march=rv64gc_zicfilp0p4_zicfiss0p4 -c -o - %s | llvm-readobj -n - | FileCheck -check-prefix=CHECK -check-prefix=CHECK_ZICFILP_ZICFISS %s


// CHECK: Name: .note.gnu.property
// CHECK: Type: NT_GNU_PROPERTY_TYPE_0
// CHECK: Property [
// CHECK_ZICFISS: riscv feature: ZICFISS
// CHECK_ZICFILP: riscv feature: ZICFILP
// CHECK_ZICFILP_ZICFISS: riscv feature: ZICFILP, ZICFISS
// CHECK: ]

// GNU Note Section Example
/*.section .note.gnu.property, "a";
.balign 8;
.long 0x4;
.long 0x10;
.long 0x5
.asciz "GNU"
.long 0xc0000000
.long 4
.long 3
.long 0*/
