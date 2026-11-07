// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 1 -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,TEXTREL
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mpic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,TEXTREL
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -pic-is-pie -mno-pic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple thumbv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple thumbv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -O0 -mllvm -fast-isel -mllvm -fast-isel-abort=1 -mllvm -arm-force-fast-isel -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -O2 -mllvm -global-isel=1 -mllvm -global-isel-abort=1 -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple armebv7-unknown-linux-gnueabihf -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang_cc1 -triple thumbv6m-unknown-none-eabi -mrelocation-model pic -pic-level 2 -mno-pic-data-is-text-relative -mllvm -fast-isel=false -S -o - %s | FileCheck %s --check-prefixes=COMMON,GOT
// RUN: %clang -target armv7-unknown-linux-gnueabihf -fPIC -mno-pic-data-is-text-relative -c -o %t.o %s
// RUN: llvm-readobj --relocations --arch-specific %t.o | FileCheck %s --check-prefix=OBJ
// RUN: %clang -target armv7-unknown-linux-gnueabihf -fPIE -mno-pic-data-is-text-relative -c -o %t.pie.o %s
// RUN: llvm-readobj --relocations --arch-specific %t.pie.o | FileCheck %s --check-prefix=OBJ

// TEXTREL: .eabi_attribute 15, 1
// TEXTREL: .eabi_attribute 14, 0

// GOT: .eabi_attribute 15, 2
// GOT: .eabi_attribute 14, 1

// OBJ: R_ARM_GOT_BREL local_data
// OBJ: R_ARM_GOT_BREL local_const
// OBJ: R_ARM_GOT_BREL external_data
// OBJ: R_ARM_GOT_BREL hidden_data
// OBJ: R_ARM_GOT_BREL weak_data
// OBJ-NOT: R_ARM_GOT_BREL local_function
// OBJ: TagName: ABI_PCS_R9_use
// OBJ: Description: Static Base
// OBJ: TagName: ABI_PCS_RW_data
// OBJ: Description: SB-relative

static int local_data;
static const int local_const = 1;
extern int external_data;
__attribute__((visibility("hidden"))) extern int hidden_data;
__attribute__((weak)) int weak_data;

static int local_function(void) { return 0; }

int *local_data_addr(void) { return &local_data; }
const int *local_const_addr(void) { return &local_const; }
int *external_data_addr(void) { return &external_data; }
int *hidden_data_addr(void) { return &hidden_data; }
int *weak_data_addr(void) { return &weak_data; }
int (*local_function_addr(void))(void) { return local_function; }

// TEXTREL-LABEL: local_data_addr:
// TEXTREL: .long local_data-(
// TEXTREL-NOT: local_data(GOT)
// TEXTREL-NOT: local_data(GOT_PREL)

// GOT-LABEL: local_data_addr:
// GOT: {{.*}}r9
// GOT: .long local_data(GOT)
// GOT-NOT: local_data(GOT_PREL)

// TEXTREL-LABEL: local_const_addr:
// TEXTREL: .long local_const-(
// TEXTREL-NOT: local_const(GOT)
// TEXTREL-NOT: local_const(GOT_PREL)

// GOT-LABEL: local_const_addr:
// GOT: {{.*}}r9
// GOT: .long local_const(GOT)
// GOT-NOT: local_const(GOT_PREL)

// COMMON-LABEL: external_data_addr:
// TEXTREL: .long external_data(GOT_PREL)-(
// GOT: {{.*}}r9
// GOT: .long external_data(GOT)

// TEXTREL-LABEL: hidden_data_addr:
// TEXTREL: .long hidden_data-(
// TEXTREL-NOT: hidden_data(GOT)
// TEXTREL-NOT: hidden_data(GOT_PREL)

// GOT-LABEL: hidden_data_addr:
// GOT: {{.*}}r9
// GOT: .long hidden_data(GOT)
// GOT-NOT: hidden_data(GOT_PREL)

// COMMON-LABEL: weak_data_addr:
// TEXTREL: .long weak_data(GOT_PREL)-(
// GOT: {{.*}}r9
// GOT: .long weak_data(GOT)

// GOT-LABEL: local_function_addr:
// GOT: .long local_function-(
// GOT-NOT: local_function(GOT_PREL)
