// Use .spv when the requested output is SPIR-V.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=amdgcnspirv --offload-device-only --no-gpu-bundle-output \
// RUN:   -nogpuinc -nogpulib -x hip %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPIRV %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=amdgcnspirv --offload-device-only --no-gpu-bundle-output \
// RUN:   -nogpuinc -nogpulib -c -x hip %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPIRV %s

// SPIRV: "-triple=spirv64-amd-amdhsa" "-emit-obj"{{.*}} "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.spv"
// SPIRV-NOT: "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.out"

// -c -emit-llvm emits a SPIR-V module, and we make the extension match that.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=amdgcnspirv --offload-device-only --no-gpu-bundle-output \
// RUN:   -nogpuinc -nogpulib -c -emit-llvm -x hip %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPIRV-EMIT-LLVM %s

// SPIRV-EMIT-LLVM: "-triple=spirv64-amd-amdhsa" "-emit-obj"{{.*}} "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.spv"
// SPIRV-EMIT-LLVM-NOT: "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.out"

// The translator path uses the same filename.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=amdgcnspirv --offload-device-only --no-gpu-bundle-output \
// RUN:   -nogpuinc -nogpulib -no-use-spirv-backend -x hip %s 2>&1 \
// RUN: | FileCheck -check-prefix=TRANSLATOR %s

// TRANSLATOR: "llvm-spirv"{{.*}} "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.spv"
// TRANSLATOR-NOT: "-o" "hip-amdgcnspirv-output-name-hip-spirv64-amd-amdhsa-amdgcnspirv.out"
