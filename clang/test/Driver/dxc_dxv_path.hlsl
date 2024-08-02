// RUN: %clang_dxc -I test -Tlib_6_3  -### %s 2>&1 | FileCheck %s

// Make sure report warning.
// CHECK:dxv not found

// RUN: echo "dxv" > %T/dxv && chmod 754 %T/dxv && %clang_dxc --dxv-path=%T %s -Tlib_6_3 -### 2>&1 | FileCheck %s --check-prefix=DXV_PATH
// DXV_PATH:dxv{{(.exe)?}}" "-" "-o" "-"

// RUN: %clang_dxc -I test -Vd -Tlib_6_3  -### %s 2>&1 | FileCheck %s --check-prefix=VD
// VD:"-cc1"{{.*}}"-triple" "dxilv1.3-unknown-shadermodel6.3-library"
// VD-NOT:dxv not found

// RUN: %clang_dxc -Tlib_6_3 -ccc-print-bindings --dxv-path=%T -Fo %t.dxo  %s 2>&1 | FileCheck %s --check-prefix=BINDINGS
// BINDINGS: "dxilv1.3-unknown-shadermodel6.3-library" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[DXC:.+]].dxo"
// BINDINGS-NEXT: "dxilv1.3-unknown-shadermodel6.3-library" - "hlsl::Validator", inputs: ["[[DXC]].dxo"]

// RUN: %clang_dxc -Tlib_6_3 -ccc-print-phases --dxv-path=%T -Fo %t.dxc  %s 2>&1 | FileCheck %s --check-prefix=PHASES

// PHASES: 0: input, "[[INPUT:.+]]", hlsl
// PHASES-NEXT: 1: preprocessor, {0}, c++-cpp-output
// PHASES-NEXT: 2: compiler, {1}, ir
// PHASES-NEXT: 3: backend, {2}, assembler
// PHASES-NEXT: 4: assembler, {3}, object
// PHASES-NEXT: 5: binary-analyzer, {4}, dx-container
