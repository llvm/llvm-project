// REQUIRES: x86-registered-target

// RUN: not %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -fsplit-lto-unit -emit-llvm < %s 2>&1 | FileCheck %s --check-prefixes=NO-UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -funified-lto -ffat-lto-objects -fsplit-lto-unit -emit-llvm < %s  | FileCheck %s --check-prefixes=FULL,SPLIT,UNIFIED
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -funified-lto -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=FULL,NOSPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -funified-lto -fsplit-lto-unit -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=THIN,SPLIT,UNIFIED
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -funified-lto -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=THIN,NOSPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -funified-lto -ffat-lto-objects -fsplit-lto-unit -emit-obj < %s -o %t.full.split.o
// RUN: llvm-readelf -S %t.full.split.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.split.bc %t.full.split.o
// RUN: llvm-dis %t.full.split.bc -o - | FileCheck %s --check-prefixes=THIN,SPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -funified-lto -ffat-lto-objects -emit-obj < %s -o %t.full.nosplit.o
// RUN: llvm-readelf -S %t.full.nosplit.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.nosplit.bc %t.full.nosplit.o
// RUN: llvm-dis %t.full.nosplit.bc -o - | FileCheck %s --check-prefixes=THIN,NOSPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -funified-lto -fsplit-lto-unit -ffat-lto-objects -emit-obj < %s -o %t.thin.split.o
// RUN: llvm-readelf -S %t.thin.split.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.split.bc %t.thin.split.o
// RUN: llvm-dis %t.thin.split.bc -o - | FileCheck %s --check-prefixes=THIN,SPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -funified-lto -ffat-lto-objects -emit-obj < %s -o %t.thin.nosplit.o
// RUN: llvm-readelf -S %t.thin.nosplit.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.nosplit.bc %t.thin.nosplit.o
// RUN: llvm-dis %t.thin.nosplit.bc -o -  | FileCheck %s --check-prefixes=THIN,NOSPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -funified-lto -ffat-lto-objects -fsplit-lto-unit -S < %s -o - \
// RUN: | FileCheck %s --check-prefixes=ASM

/// Be sure we enable split LTO units correctly under -ffat-lto-objects.
//   SPLIT: ![[#]] = !{i32 1, !"EnableSplitLTOUnit", i32 1}
// NOSPLIT: ![[#]] = !{i32 1, !"EnableSplitLTOUnit", i32 0}

// FULL-NOT: ![[#]] = !{i32 1, !"ThinLTO", i32 0}
// THIN-NOT: ![[#]] = !{i32 1, !"ThinLTO", i32 0}

/// FatLTO always uses UnifiedLTO. It's an error if they aren't set together
//    UNIFIED: ![[#]] = !{i32 1, !"UnifiedLTO", i32 1}
// NO-UNIFIED: error: invalid argument '-ffat-lto-objects' only allowed with '-funified-lto'

// ELF: .llvm.lto

//      ASM: .section        .llvm.lto,"e",@progbits
// ASM-NEXT: .Lllvm.embedded.object:
// ASM-NEXT:        .asciz  "BC
// ASM-NEXT: .size   .Lllvm.embedded.object

int test(void) {
  return 0xabcd;
}
