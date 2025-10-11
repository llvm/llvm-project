// RUN: %clang -### -target x86_64 -c -gdwarf -gkey-instructions %s 2>&1 | FileCheck %s --check-prefixes=KEY-INSTRUCTIONS
// RUN: %clang -### -target x86_64 -c -gdwarf -gno-key-instructions %s 2>&1 | FileCheck %s --check-prefixes=NO-KEY-INSTRUCTIONS
// RUN: %clang -### -target x86_64 -c -gno-key-instructions %s 2>&1 | FileCheck %s --check-prefixes=NO-DEBUG

//// Help.
// RUN %clang --help | FileCheck %s --check-prefix=HELP
// HELP: -gkey-instructions  Enable Key Instructions, which reduces the jumpiness of debug stepping in optimized C/C++ code in some debuggers. DWARF only.

// KEY-INSTRUCTIONS: "-gkey-instructions"
// NO-KEY-INSTRUCTIONS-NOT: key-instructions
// NO-DEBUG-NOT: debug-info-kind
// NO-DEBUG-NOT: dwarf

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

//// Enable Key Instructions by default if optimisations are enabled and we're
//// emitting DWARF.
////
//// | opt level | -gkey-instructions | feature |
//// |         0 |                 no |     off |
//// |         0 |                yes |      on |
//// |       >=1 |                 no |      on |
//// |       >=1 |                yes |      on |
//// |       >=1 |   no & no -g flags |     off |
//// |       >=1 | no & emit codeview |     off |
//
// RUN: %clang %s     -target x86_64 -gdwarf    -gmlt                    -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang %s     -target x86_64 -gdwarf    -gmlt -gkey-instructions -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang %s -O0 -target x86_64 -gdwarf    -gmlt                    -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang %s -O0 -target x86_64 -gdwarf    -gmlt -gkey-instructions -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang %s -O1 -target x86_64 -gdwarf    -gmlt                    -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang %s -O1 -target x86_64 -gdwarf    -gmlt -gkey-instructions -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang %s -O1 -target x86_64                                     -### 2>&1 | FileCheck %s --check-prefixes=NO-KEY-INSTRUCTIONS
// RUN: %clang %s -O1 -target x86_64 -gcodeview -gmlt                    -### 2>&1 | FileCheck %s --check-prefixes=NO-KEY-INSTRUCTIONS
//
// RUN: %clang %s     -target x86_64 -gdwarf    -gmlt                    -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-OFF
// RUN: %clang %s     -target x86_64 -gdwarf    -gmlt -gkey-instructions -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-ON
// RUN: %clang %s -O0 -target x86_64 -gdwarf    -gmlt                    -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-OFF
// RUN: %clang %s -O0 -target x86_64 -gdwarf    -gmlt -gkey-instructions -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-ON
// RUN: %clang %s -O1 -target x86_64 -gdwarf    -gmlt                    -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-ON
// RUN: %clang %s -O1 -target x86_64 -gdwarf    -gmlt -gkey-instructions -S -emit-llvm -o - | FileCheck %s --check-prefix=SMOKETEST-ON
// RUN: %clang %s -O1 -target x86_64                                     -S -emit-llvm -o - | FileCheck %s --check-prefixes=SMOKETEST-OFF,SMOKETEST-NO-DEBUG
// RUN: %clang %s -O1 -target x86_64 -gcodeview -gmlt                    -S -emit-llvm -o - | FileCheck %s --check-prefixes=SMOKETEST-OFF
// SMOKETEST-NO-DEBUG: llvm.module.flags
// SMOKETEST-NO-DEBUG-NOT: DICompileUnit

//// Check only "plain" C/C++ turns on Key Instructions by default.
// RUN: %clang -x c             %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang -x c++           %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=KEY-INSTRUCTIONS
// RUN: %clang -x cuda -nocudalib -nocudainc %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang -x hip  -nogpulib  -nogpuinc  %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang -x cl            %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang -x objective-c   %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
// RUN: %clang -x objective-c++ %s -O1 -target x86_64 -gdwarf -gmlt -### 2>&1 | FileCheck %s --check-prefix=NO-KEY-INSTRUCTIONS
