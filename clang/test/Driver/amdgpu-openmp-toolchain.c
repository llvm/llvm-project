// REQUIRES: amdgpu-registered-target
// RUN:   env LIBRARY_PATH=%S/Inputs/hip_dev_lib %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "c"{{.*}}
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "ir"{{.*}}
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906" "-fcuda-is-device" "-emit-llvm-bc" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgcn-gfx906.bc"{{.*}}
// CHECK: llvm-link{{.*}}"-o" "{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-linked-{{.*}}.bc"
// CHECK: llc{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-linked-{{.*}}.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-{{.*}}.o"
// CHECK: lld{{.*}}"-flavor" "gnu" "--no-undefined" "-shared" "-o"{{.*}}amdgpu-openmp-toolchain-{{.*}}.out" "{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-{{.*}}.o"
// CHECK: clang-offload-wrapper{{.*}}"-target" "x86_64-unknown-linux-gnu" "-o" "{{.*}}a-{{.*}}.bc" {{.*}}amdgpu-openmp-toolchain-{{.*}}.out"
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-o" "{{.*}}a-{{.*}}.o" "-x" "ir" "{{.*}}a-{{.*}}.bc"
// CHECK: ld{{.*}}"-o" "a.out"{{.*}}"{{.*}}amdgpu-openmp-toolchain-{{.*}}.o" "{{.*}}a-{{.*}}.o" "-lomp" "-lomptarget"

// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PHASES %s
// phases
// CHECK-PHASES: 0: input, "{{.*}}amdgpu-openmp-toolchain.c", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: backend, {2}, assembler, (host-openmp)
// CHECK-PHASES: 4: assembler, {3}, object, (host-openmp)
// CHECK-PHASES: 5: input, "{{.*}}amdgpu-openmp-toolchain.c", c, (device-openmp)
// CHECK-PHASES: 6: preprocessor, {5}, cpp-output, (device-openmp)
// CHECK-PHASES: 7: compiler, {6}, ir, (device-openmp)
// CHECK-PHASES: 8: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (amdgcn-amd-amdhsa)" {7}, ir
// CHECK-PHASES: 9: backend, {8}, assembler, (device-openmp)
// CHECK-PHASES: 10: assembler, {9}, object, (device-openmp)
// CHECK-PHASES: 11: linker, {10}, image, (device-openmp)
// CHECK-PHASES: 12: offload, "device-openmp (amdgcn-amd-amdhsa)" {11}, image
// CHECK-PHASES: 13: clang-offload-wrapper, {12}, ir, (host-openmp)
// CHECK-PHASES: 14: backend, {13}, assembler, (host-openmp)
// CHECK-PHASES: 15: assembler, {14}, object, (host-openmp)
// CHECK-PHASES: 16: linker, {4, 15}, image, (host-openmp)

// handling of --libomptarget-amdgcn-bc-path
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --libomptarget-amdgcn-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgcn-gfx803.bc %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIBOMPTARGET
// CHECK-LIBOMPTARGET: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-emit-llvm-bc" "-mlink-builtin-bitcode"{{.*}}Inputs/hip_dev_lib/libomptarget-amdgcn-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOGPULIB
// CHECK-NOGPULIB-NOT: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-emit-llvm-bc" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgcn-gfx803.bc"{{.*}}
