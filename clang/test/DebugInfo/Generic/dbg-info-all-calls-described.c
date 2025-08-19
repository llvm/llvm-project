// Test that call site debug info is (un)supported in various configurations.

// Supported: DWARF5, -O1, standalone DI
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -O1 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone -dwarf-version=5 \
// RUN: | FileCheck %s -check-prefix=HAS-ATTR \
// RUN:     -implicit-check-not=DISubprogram -implicit-check-not=DIFlagAllCallsDescribed

// Supported: DWARF4 + LLDB tuning, -O1, limited DI
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -O1 -disable-llvm-passes \
// RUN:   -debugger-tuning=lldb \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=HAS-ATTR \
// RUN:     -implicit-check-not=DISubprogram -implicit-check-not=DIFlagAllCallsDescribed

// Note: DIFlagAllCallsDescribed may have been enabled prematurely when tuning
// for GDB under -gdwarf-4 in https://reviews.llvm.org/D69743. It's possible
// this should have been 'Unsupported' until entry values emission was enabled
// by default.
//
// Supported: DWARF4 + GDB tuning
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu \
// RUN:   %s -o - -O1 -disable-llvm-passes -debugger-tuning=gdb \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=HAS-ATTR \
// RUN:     -implicit-check-not=DIFlagAllCallsDescribed

// Supported: DWARF4 + LLDB, -O1
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu \
// RUN:   %s -o - -O1 -disable-llvm-passes -debugger-tuning=lldb \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=HAS-ATTR \
// RUN:     -implicit-check-not=DIFlagAllCallsDescribed

// Unsupported: -O0
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu \
// RUN:   %s -o - -O0 -disable-llvm-passes -debugger-tuning=gdb \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=NO-ATTR

// Supported: DWARF4 + LLDB tuning, -O1, line-tables only DI
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -O1 -disable-llvm-passes \
// RUN:   -debugger-tuning=lldb \
// RUN:   -debug-info-kind=line-tables-only -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=LINE-TABLES-ONLY

// Unsupported: -O0
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -O0 \
// RUN:   -debug-info-kind=standalone -dwarf-version=5 \
// RUN: | FileCheck %s -check-prefix=NO-ATTR

// Unsupported: DWARF4
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - \
// RUN:   -O1 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 \
// RUN: | FileCheck %s -check-prefix=NO-ATTR

// NO-ATTR-NOT: FlagAllCallsDescribed

// HAS-ATTR-DAG: DISubprogram(name: "declaration1", {{.*}}, spFlags: DISPFlagOptimized)
// HAS-ATTR-DAG: DISubprogram(name: "declaration2", {{.*}}, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized 
// HAS-ATTR-DAG: DISubprogram(name: "declaration3", {{.*}}, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
// HAS-ATTR-DAG: DISubprogram(name: "declaration4", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized

// HAS-ATTR-DAG: DISubprogram(name: "force_irgen", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition

// LINE-TABLES-ONLY: DISubprogram(name: "force_irgen", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition

void declaration1();

void declaration2();

void declaration2() {}

void declaration3(void);

void declaration4(void);

void declaration4(void) {}

void __attribute__((optnone)) force_irgen(void) {
  declaration1();
  declaration3();
}
