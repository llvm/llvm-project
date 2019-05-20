// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=0.0 %s -o - 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-0-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-0-0: the clang compiler does not support '-mamdgpu-debugger-abi=0.0'

// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=1.0 %s -o - 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-1-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-1-0: the clang compiler does not support '-mamdgpu-debugger-abi=1.0'

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -mcode-object-v3 %s 2>&1 | FileCheck --check-prefix=CODE-OBJECT-V3 %s
// CODE-OBJECT-V3: "-target-feature" "+code-object-v3"

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -mno-code-object-v3 %s 2>&1 | FileCheck --check-prefix=NO-CODE-OBJECT-V3 %s
// NO-CODE-OBJECT-V3: "-target-feature" "-code-object-v3"

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -mxnack %s 2>&1 | FileCheck --check-prefix=XNACK %s
// XNACK: "-target-feature" "+xnack"

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -mno-xnack %s 2>&1 | FileCheck --check-prefix=NO-XNACK %s
// NO-XNACK: "-target-feature" "-xnack"

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -msram-ecc %s 2>&1 | FileCheck --check-prefix=SRAM-ECC %s
// SRAM-ECC: "-target-feature" "+sram-ecc"

// RUN: %clang -### -target amdgcn -mcpu=gfx700 -mno-sram-ecc %s 2>&1 | FileCheck --check-prefix=NO-SRAM-ECC %s
// NO-SRAM-ECC: "-target-feature" "-sram-ecc"
