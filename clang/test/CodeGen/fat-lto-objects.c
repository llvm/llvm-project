// REQUIRES: x86-registered-target

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -fsplit-lto-unit -emit-llvm < %s  | FileCheck %s --check-prefixes=FULL,SPLIT
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=FULL,SPLIT

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -fsplit-lto-unit -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=THIN,SPLIT
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -ffat-lto-objects -emit-llvm < %s  | FileCheck %s --check-prefixes=THIN,NOSPLIT

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -fsplit-lto-unit -emit-obj < %s -o %t.full.split.o
// RUN: llvm-readelf -S %t.full.split.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.split.bc %t.full.split.o
// RUN: llvm-dis %t.full.split.bc -o - | FileCheck %s --check-prefixes=FULL,SPLIT,NOUNIFIED

/// Full LTO always sets EnableSplitLTOUnit when the summary is used.
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -emit-obj < %s -o %t.full.nosplit.o
// RUN: llvm-readelf -S %t.full.nosplit.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.full.nosplit.bc %t.full.nosplit.o
// RUN: llvm-dis %t.full.nosplit.bc -o - | FileCheck %s --check-prefixes=FULL,SPLIT,NOUNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -fsplit-lto-unit -ffat-lto-objects -emit-obj < %s -o %t.thin.split.o
// RUN: llvm-readelf -S %t.thin.split.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.split.bc %t.thin.split.o
// RUN: llvm-dis %t.thin.split.bc -o - | FileCheck %s --check-prefixes=THIN,SPLIT,NOUNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -ffat-lto-objects -emit-obj < %s -o %t.thin.nosplit.o
// RUN: llvm-readelf -S %t.thin.nosplit.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.thin.nosplit.bc %t.thin.nosplit.o
// RUN: llvm-dis %t.thin.nosplit.bc -o -  | FileCheck %s --check-prefixes=THIN,NOSPLIT,NOUNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=thin -funified-lto -ffat-lto-objects -emit-obj < %s -o %t.unified.o
// RUN: llvm-readelf -S %t.unified.o | FileCheck %s --check-prefixes=ELF
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.unified.bc %t.unified.o
// RUN: llvm-dis %t.unified.bc -o - | FileCheck %s --check-prefixes=THIN,NOSPLIT,UNIFIED

// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -flto=full -ffat-lto-objects -fsplit-lto-unit -S < %s -o - \
// RUN: | FileCheck %s --check-prefixes=ASM

/// Make sure that FatLTO generates .llvm.lto sections that are the same as the output from normal LTO compilations
// RUN: %clang -O2 --target=x86_64-unknown-linux-gnu -fPIE -flto=full -ffat-lto-objects -c %s -o %t.fatlto.full.o
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.fatlto.full.bc %t.fatlto.full.o
// RUN: llvm-dis < %t.fatlto.full.bc -o %t.fatlto.full.ll
// RUN: %clang -O2 --target=x86_64-unknown-linux-gnu -fPIE -flto=full -c %s -o %t.nofat.full.bc
// RUN: llvm-dis < %t.nofat.full.bc -o %t.nofat.full.ll
// RUN: diff %t.fatlto.full.ll %t.nofat.full.ll

// RUN: %clang -O2 --target=x86_64-unknown-linux-gnu -fPIE -flto=thin -ffat-lto-objects -c %s -o %t.fatlto.thin.o
// RUN: llvm-objcopy --dump-section=.llvm.lto=%t.fatlto.thin.bc %t.fatlto.thin.o
// RUN: llvm-dis < %t.fatlto.thin.bc -o %t.fatlto.thin.ll
// RUN: %clang -O2 --target=x86_64-unknown-linux-gnu -fPIE -flto=thin -c %s -o %t.nofat.thin.bc
// RUN: llvm-dis < %t.nofat.thin.bc -o %t.nofat.thin.ll
// RUN: diff %t.fatlto.thin.ll %t.nofat.thin.ll

/// Be sure we enable split LTO units correctly under -ffat-lto-objects.
//   SPLIT: ![[#]] = !{i32 1, !"EnableSplitLTOUnit", i32 1}
// NOSPLIT: ![[#]] = !{i32 1, !"EnableSplitLTOUnit", i32 0}

// FULL-NOT: ![[#]] = !{i32 1, !"ThinLTO", i32 0}
// THIN-NOT: ![[#]] = !{i32 1, !"ThinLTO", i32 0}

// UNIFIED: ![[#]] = !{i32 1, !"UnifiedLTO", i32 1}
// NOUNIFIED-NOT: ![[#]] = !{i32 1, !"UnifiedLTO", i32 1}

// ELF: .llvm.lto

//      ASM: .section        .llvm.lto,"e",@progbits
// ASM-NEXT: .Lllvm.embedded.object:
// ASM-NEXT:        .asciz  "BC
// ASM-NEXT: .size   .Lllvm.embedded.object

const char* foo = "foo";

int test(void) {
  const char* bar = "bar";
  return 0xabcd;
}
