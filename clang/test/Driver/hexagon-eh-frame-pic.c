/// Hexagon selects the .eh_frame FDE pointer encoding from whether the code is
/// position independent (DW_EH_PE_pcrel vs DW_EH_PE_absptr, see
/// MCObjectFileInfo::initELFMCObjectFileInfo).  llvm-mc defaults to non-PIC, so
/// the driver has to tell it, or -fno-integrated-as produces unwind tables that
/// disagree with the ones the integrated assembler produces for the same input
/// (R_HEX_32 instead of R_HEX_32_PCREL).

/// hexagon-unknown-linux-musl defaults to PIC.
// RUN: %clang --target=hexagon-unknown-linux-musl -fno-integrated-as \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=PIC

// RUN: %clang --target=hexagon-unknown-linux-musl -fno-integrated-as -fPIC \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=PIC

// RUN: %clang --target=hexagon-unknown-linux-musl -fno-integrated-as -fpie \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=PIC

/// Non-PIC keeps DW_EH_PE_absptr, so llvm-mc must not be told otherwise.
// RUN: %clang --target=hexagon-unknown-linux-musl -fno-integrated-as -fno-pic \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=NOPIC

/// Bare-metal ELF defaults to non-PIC.
// RUN: %clang --target=hexagon-unknown-elf -fno-integrated-as \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=NOPIC

// RUN: %clang --target=hexagon-unknown-elf -fno-integrated-as -fPIC \
// RUN:   -funwind-tables -c %s -### 2>&1 | FileCheck %s --check-prefix=PIC

// PIC: llvm-mc
// PIC: "-position-independent"

// NOPIC: llvm-mc
// NOPIC-NOT: "-position-independent"

void f(void);
void g(void) { f(); }
