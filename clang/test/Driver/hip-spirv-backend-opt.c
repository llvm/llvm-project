// COM: This test case validates the behavior of -hip-use-experimental-spirv-backend

// COM: Test that -hip-use-experimental-spirv-backend calls clang -cc1 with the SPIRV triple.
// RUN: %clang -x hip %s --cuda-device-only --offload-arch=amdgcnspirv -hip-use-experimental-spirv-backend -nogpuinc -nogpulib -### 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-BACKEND
// CHECK-SPIRV-BACKEND: "{{.*}}clang{{.*}}" "-cc1" "{{.*-triple=spirv64-amd-amdhsa}}"

// COM: Test that -no-hip-use-experimental-spirv-backend calls the SPIRV translator
// RUN: %clang -x hip %s --cuda-device-only --offload-arch=amdgcnspirv -no-hip-use-experimental-spirv-backend -nogpuinc -nogpulib -### 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-TRANSLATOR
// CHECK-SPIRV-TRANSLATOR: "{{.*llvm-spirv.*}}" "{{--spirv-max-version=[0-9]+\.[0-9]}}"

// COM: Test that by default we use the translator
// RUN: %clang -x hip %s --cuda-device-only --offload-arch=amdgcnspirv -nogpuinc -nogpulib -### 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-TRANSLATOR
