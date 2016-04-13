// Check handling of AMDGPU target features.
//
// -mamdgpu-debugger-abi=0.0
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=0.0 %s -o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-0-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-0-0: the clang compiler does not support '-mamdgpu-debugger-abi=0.0'
//
// -mamdgpu-debugger-abi=1.0
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=1.0 %s -o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-1-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-1-0: "-target-feature" "+amdgpu-debugger-insert-nops" "-target-feature" "+amdgpu-debugger-reserve-trap-regs"
//
// -mamdgpu-debugger-insert-nops
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-insert-nops %s -o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-INSERT-NOPS %s
// CHECK-MAMDGPU-DEBUGGER-INSERT-NOPS: "-target-feature" "+amdgpu-debugger-insert-nops"
//
// -mamdgpu-debugger-reserve-trap-regs
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-reserve-trap-regs %s -o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-RESERVE-TRAP-REGS %s
// CHECK-MAMDGPU-DEBUGGER-RESERVE-TRAP-REGS: "-target-feature" "+amdgpu-debugger-reserve-trap-regs"
