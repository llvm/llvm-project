// This test case validates the behavior of -use-spirv-backend

// --offload-device-only is always set --- testing interactions with -S and -fgpu-rdc

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend --offload-device-only -S -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-TEXTUAL

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend --offload-device-only -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BINARY

// The new driver's behavior is to emit LLVM IR for --offload-device-only and -fgpu-rdc (independently of SPIR-V).
// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -### -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -S -fgpu-rdc -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-LL,CHECK-FGPU-RDC

// The new driver's behavior is to emit LLVM IR for --offload-device-only and -fgpu-rdc (independently of SPIR-V).
// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend --offload-device-only -fgpu-rdc -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC,CHECK-FGPU-RDC

// --offload-device-only is always unset --- testing interactions with -S and -fgpu-rdc

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend -S -fgpu-rdc -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC,CHECK-FGPU-RDC

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend -S -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend -fgpu-rdc -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC,CHECK-CLANG-LINKER-WRAPPER

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC,CHECK-CLANG-LINKER-WRAPPER

// RUN: %clang --no-offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -### -x hip %s -save-temps  \
// RUN:         -use-spirv-backend -no-canonical-prefixes \
// RUN: 2>&1 | FileCheck %s --check-prefixes=CHECK-SPIRV-TRANSLATOR,CHECK-SPIRV-BACKEND-BC,CHECK-SPIRV-BACKEND-BINARY-EQ-TRIPLE

// CHECK-SPIRV-TRANSLATOR-NOT: "{{.*llvm-spirv.*}}"
// CHECK-SPIRV-BACKEND-TEXTUAL: "{{.*clang(\.exe)?}}" "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}} "-S"
// CHECK-SPIRV-BACKEND-BINARY: "{{.*clang(\.exe)?}}" "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}} "-emit-obj"
// CHECK-SPIRV-BACKEND-BC: "{{.*clang(\.exe)?}}" "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}} "-emit-llvm-bc"
// CHECK-SPIRV-BACKEND-LL: "{{.*clang(\.exe)?}}" "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}} "-emit-llvm"
// CHECK-SPIRV-BACKEND-BINARY-EQ-TRIPLE: "{{.*clang(\.exe)?}}" "-cc1" {{.*}}"-triple=spirv64-amd-amdhsa" {{.*}}"-emit-obj"
// CHECK-FGPU-RDC-SAME: {{.*}} "-fgpu-rdc"
// CHECK-CLANG-LINKER-WRAPPER: "{{.*}}clang-linker-wrapper" "--should-extract=amdgcnspirv" {{.*}} "--device-compiler=spirv64-amd-amdhsa=-use-spirv-backend"
