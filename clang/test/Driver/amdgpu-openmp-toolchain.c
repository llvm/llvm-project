// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:          --offload-arch=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib -nogpulib %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-llvm-bc"{{.*}}"-x" "c"
// CHECK: "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}}"-target-cpu" "gfx906"{{.*}}"-fcuda-is-device"{{.*}}
// CHECK: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-emit-obj"
// CHECK: clang-linker-wrapper{{.*}}"--"{{.*}} "-o" "a.out"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, ir, (device-openmp)
// CHECK-PHASES: 8: offload, "device-openmp (amdgcn-amd-amdhsa)" {7}, ir
// CHECK-PHASES: 9: clang-offload-packager, {8}, image, (device-openmp)
// CHECK-PHASES: 10: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (x86_64-unknown-linux-gnu)" {9}, ir
// CHECK-PHASES: 11: backend, {10}, assembler, (host-openmp)
// CHECK-PHASES: 12: assembler, {11}, object, (host-openmp)
// CHECK-PHASES: 13: clang-linker-wrapper, {12}, image, (host-openmp)

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_BC]]"], output: "[[BINARY:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[BINARY]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -save-temps -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS-TEMPS
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -save-temps -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-BINDINGS-TEMPS
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_PP:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_PP]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS-TEMPS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[DEVICE_PP:.+]]"
// CHECK-BINDINGS-TEMPS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[DEVICE_PP]]", "[[HOST_BC]]"], output: "[[DEVICE_TEMP_BC:.+]]"
// CHECK-BINDINGS-TEMPS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[DEVICE_TEMP_BC]]"], output: "[[DEVICE_BC:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "Offload::Packager", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_IMAGE:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]", "[[DEVICE_IMAGE]]"], output: "[[HOST_ASM:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "clang::as", inputs: ["[[HOST_ASM]]"], output: "[[HOST_OBJ:.+]]"
// CHECK-BINDINGS-TEMPS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx803 \
// RUN:   --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s  2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-LIB-DEVICE
// CHECK-LIB-DEVICE: "-cc1" {{.*}}ocml.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx803 -nogpulib \
// RUN:   --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s  2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-LIB-DEVICE-NOGPULIB
// CHECK-LIB-DEVICE-NOGPULIB-NOT: "-cc1" {{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx90a:sramecc-:xnack+ \
// RUN:   -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-TARGET-ID
// CHECK-TARGET-ID: clang-offload-packager{{.*}}arch=gfx90a:sramecc-:xnack+,kind=openmp,feature=-sramecc,feature=+xnack

// RUN: not %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx90a,gfx90a:xnack+ \
// RUN:   -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-TARGET-ID-ERROR
// CHECK-TARGET-ID-ERROR: error: invalid offload arch combinations: 'gfx90a' and 'gfx90a:xnack+'

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx90a \
// RUN:   -O3 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-OPT
// CHECK-OPT: clang-linker-wrapper{{.*}}"--opt-level=O3"
