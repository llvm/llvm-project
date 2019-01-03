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

// HAS-ATTR-DAG: DISubprogram(name: "declaration2", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition
// HAS-ATTR-DAG: DISubprogram(name: "struct1", {{.*}}, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
// HAS-ATTR-DAG: DISubprogram(name: "struct1", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition
// HAS-ATTR-DAG: DISubprogram(name: "method1", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition
// HAS-ATTR-DAG: DISubprogram(name: "force_irgen", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition

// LINE-TABLES-ONLY: DISubprogram(name: "force_irgen", {{.*}}, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition

void declaration1();

void declaration2();

void declaration2() {}

struct struct1 {
  struct1() {}
  void method1() {}
};

void __attribute__((optnone)) force_irgen() {
  declaration1();
  struct1().method1();
}
