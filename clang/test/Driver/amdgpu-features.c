// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=0.0 %s -o - 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-0-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-0-0: the clang compiler does not support '-mamdgpu-debugger-abi=0.0'

// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri -mamdgpu-debugger-abi=1.0 %s -o - 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MAMDGPU-DEBUGGER-ABI-1-0 %s
// CHECK-MAMDGPU-DEBUGGER-ABI-1-0: "-target-feature" "+amdgpu-debugger-insert-nops" "-target-feature" "+amdgpu-debugger-reserve-regs" "-target-feature" "+amdgpu-debugger-emit-prologue"
