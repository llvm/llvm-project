// With no output args, we emit assembly to stdout.
// RUN: %clang_dxc -T lib_6_7 -Vd %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-STDOUT
// RUN: %clang_dxc -T lib_6_7 -Vd %s -ccc-print-phases 2>&1 | FileCheck %s --check-prefixes=CHECK-PHASES,CHECK-PHASES-ASM

// Same if we explicitly ask for assembly (-Fc) to stdout.
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc - -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-STDOUT
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc - -ccc-print-phases 2>&1 | FileCheck %s --check-prefixes=CHECK-PHASES,CHECK-PHASES-ASM

// DXIL Assembly to a file.
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc x.asm -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-ASM
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fcx.asm -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-ASM
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc x.asm -ccc-print-phases 2>&1 | FileCheck %s --check-prefixes=CHECK-PHASES,CHECK-PHASES-ASM

// DXIL Object code to a file.
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fo x.obj -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-OBJ
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fox.obj -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-OBJ
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fo x.obj -ccc-print-phases 2>&1 | FileCheck %s --check-prefixes=CHECK-PHASES,CHECK-PHASES-OBJ

// If both -Fc and -Fo are provided, we generate both files.
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc x.asm -Fo x.obj -### 2>&1 | FileCheck %s --check-prefixes=CHECK-CC1,CHECK-CC1-ASM,CHECK-CC1-BOTH
// RUN: %clang_dxc -T lib_6_7 -Vd %s -Fc x.asm -Fo x.obj -ccc-print-phases 2>&1 | FileCheck %s --check-prefixes=CHECK-PHASES,CHECK-PHASES-OBJ

// CHECK-PHASES:         0: input, {{.*}}, hlsl
// CHECK-PHASES:         1: preprocessor, {0}, c++-cpp-output
// CHECK-PHASES:         2: compiler, {1}, ir
// CHECK-PHASES:         3: backend, {2}, assembler
// CHECK-PHASES-ASM-NOT: 4: assembler, {3}, object
// CHECK-PHASES-OBJ:     4: assembler, {3}, object

// CHECK-CC1: "-cc1"
// CHECK-CC1-STDOUT-SAME: "-o" "-"

// CHECK-CC1-ASM-SAME: "-S"
// CHECK-CC1-ASM-SAME: "-o" "x.asm"

// CHECK-CC1-OBJ-SAME: "-emit-obj"
// CHECK-CC1-OBJ-SAME: "-o" "x.obj"

// For the case where we specify both -Fc and -Fo, we emit the asm as part of
// cc1 and invoke cc1as for the object.
// CHECK-CC1-BOTH: "-cc1as"
// CHECK-CC1-BOTH-SAME: "-o" "x.obj"
