// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar --hipstdpar-path=/does/not/exist -nogpulib    \
// RUN:   -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-MISSING-LIB %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar --hipstdpar-path=%S/Inputs/hipstdpar             \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   -nogpulib -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-COMPILE %s
// RUN: touch %t.o
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --hipstdpar %t.o 2>&1 | FileCheck --check-prefix=HIPSTDPAR-LINK %s

// HIPSTDPAR-MISSING-LIB: error: cannot find HIP Standard Parallelism Acceleration library; provide it via '--hipstdpar-path'
// HIPSTDPAR-COMPILE: "-x" "hip"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/thrust}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/rocprim}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/Inputs/hipstdpar}}"
// HIPSTDPAR-COMPILE: "-include" "hipstdpar_lib.hpp"
// HIPSTDPAR-LINK: "-rpath"
// HIPSTDPAR-LINK: "{{.*hip.*}}"

// Check that --hipstdpar is forwarded to the linker wrapper as a device
// compiler arg when using the new offloading driver.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar --hipstdpar-path=%S/Inputs/hipstdpar \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   -nogpulib -nogpuinc -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-NEW-DRV %s
// HIPSTDPAR-NEW-DRV: {{".*clang-linker-wrapper"}}
// HIPSTDPAR-NEW-DRV-SAME: "--device-compiler=amdgcn-amd-amdhsa=--hipstdpar"

// Check that the base AMDGPU toolchain translates --hipstdpar to the backend
// flag. This is the path taken by the inner clang invocation from the linker
// wrapper (clang --target=amdgcn-amd-amdhsa --hipstdpar ...).
// RUN: %clang -### --target=amdgcn-amd-amdhsa \
// RUN:   --hipstdpar --hipstdpar-path=%S/Inputs/hipstdpar \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-AMDGPU-TC %s
// HIPSTDPAR-AMDGPU-TC: "-mllvm" "-amdgpu-enable-hipstdpar"

// Check that the base AMDGPU toolchain linker forwards the hipstdpar flag as a
// plugin-opt for the LTO path.
// RUN: %clang -### --target=amdgcn-amd-amdhsa \
// RUN:   --hipstdpar -flto --hipstdpar-path=%S/Inputs/hipstdpar \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-AMDGPU-LTO %s
// HIPSTDPAR-AMDGPU-LTO: {{.*}}ld.lld
// HIPSTDPAR-AMDGPU-LTO-SAME: "-plugin-opt=-amdgpu-enable-hipstdpar"

// Check that without --hipstdpar none of the backend flags are added.
// RUN: %clang -### --target=amdgcn-amd-amdhsa \
// RUN:   -flto -nogpulib %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NO-HIPSTDPAR %s
// NO-HIPSTDPAR-NOT: "-amdgpu-enable-hipstdpar"
