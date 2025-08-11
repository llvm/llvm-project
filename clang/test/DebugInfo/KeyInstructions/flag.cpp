// RUN: %clang -### -target x86_64 -c -gdwarf -gkey-instructions %s 2>&1 | FileCheck %s --check-prefixes=KEY-INSTRUCTIONS
// RUN: %clang -### -target x86_64 -c -gdwarf -gno-key-instructions %s 2>&1 | FileCheck %s --check-prefixes=NO-KEY-INSTRUCTIONS
//// Default: Off.
// RUN: %clang -### -target x86_64 -c -gdwarf %s 2>&1 | FileCheck %s --check-prefixes=NO-KEY-INSTRUCTIONS

//// Help.
// RUN %clang --help | FileCheck %s --check-prefix=HELP
// HELP: -gkey-instructions  Enable Key Instructions, which reduces the jumpiness of debug stepping in optimized C/C++ code in some debuggers. DWARF only. Implies -g.

// KEY-INSTRUCTIONS: "-gkey-instructions"
// NO-KEY-INSTRUCTIONS-NOT: key-instructions

//// Help hidden: flag should not be visible.
// RUN: %clang --help | FileCheck %s --check-prefix=HELP
// HELP-NOT: key-instructions

// Smoke test: check for Key Instructions keywords in the IR.
void f() {}
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -debug-info-kind=line-tables-only -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-OFF
// SMOKETEST-OFF-NOT: keyInstructions:
// SMOKETEST-OFF-NOT: atomGroup

// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -gkey-instructions -debug-info-kind=line-tables-only -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-ON
// SMOKETEST-ON: keyInstructions: true
// SMOKETEST-ON: atomGroup: 1
