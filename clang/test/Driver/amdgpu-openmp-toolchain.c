// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib %s 2>&1 \
// RUN:   | FileCheck %s

// verify the tools invocations
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "c"{{.*}}
// CHECK: clang{{.*}}"-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}}"-x" "ir"{{.*}}
// CHECK: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm-bc"{{.*}}"-target-cpu" "gfx906" "-fcuda-is-device"{{.*}}"-mlink-builtin-bitcode"{{.*}}libomptarget-amdgpu-gfx906.bc"{{.*}}
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

// handling of --libomptarget-amdgpu-bc-path
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIBOMPTARGET
// CHECK-LIBOMPTARGET: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}Inputs/hip_dev_lib/libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOGPULIB
// CHECK-NOGPULIB-NOT: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx803" "-fcuda-is-device" "-mlink-builtin-bitcode"{{.*}}libomptarget-amdgpu-gfx803.bc"{{.*}}

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings -save-temps -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-PRINT-BINDINGS
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.*]]"],
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang",{{.*}} output: "[[HOST_BC:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[HOST_BC]]"], output: "[[HOST_S:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang::as", inputs: ["[[HOST_S]]"], output: "[[HOST_O:.*]]"
// CHECK-PRINT-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[INPUT]]"], output: "[[DEVICE_I:.*]]"
// CHECK-PRINT-BINDINGS: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[DEVICE_I]]", "[[HOST_BC]]"], output: "[[DEVICE_BC:.*]]"
// CHECK-PRINT-BINDINGS: "amdgcn-amd-amdhsa" - "AMDGCN::OpenMPLinker", inputs: ["[[DEVICE_BC]]"], output: "[[DEVICE_OUT:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "offload wrapper", inputs: ["[[DEVICE_OUT]]"], output: "[[OFFLOAD_WRAPPER:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[OFFLOAD_WRAPPER]]"], output: "[[OFFLOAD_S:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "clang::as", inputs: ["[[OFFLOAD_S]]"], output: "[[OFFLOAD_O:.*]]"
// CHECK-PRINT-BINDINGS: "x86_64-unknown-linux-gnu" - "GNU::Linker", inputs: ["[[HOST_O]]", "[[OFFLOAD_O]]"], output:

// verify the llc is invoked for textual assembly output
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib -save-temps %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SAVE-ASM
// CHECK-SAVE-ASM: llc{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-linked.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-filetype=asm" "-o"{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906.s"
// CHECK-SAVE-ASM: llc{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906-linked.bc" "-mtriple=amdgcn-amd-amdhsa" "-mcpu=gfx906" "-filetype=obj" "-o"{{.*}}amdgpu-openmp-toolchain-{{.*}}-gfx906.o"

// check the handling of -c
// RUN:   %clang -ccc-print-bindings -c --target=x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 --libomptarget-amdgpu-bc-path=%S/Inputs/hip_dev_lib -save-temps %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-C
// CHECK-C: "x86_64-unknown-linux-gnu" - "clang",
// CHECK-C: "x86_64-unknown-linux-gnu" - "clang",{{.*}}output: "[[HOST_BC:.*]]"
// CHECK-C: "amdgcn-amd-amdhsa" - "clang",{{.*}}output: "[[DEVICE_I:.*]]"
// CHECK-C: "amdgcn-amd-amdhsa" - "clang", inputs: ["[[DEVICE_I]]", "[[HOST_BC]]"]
// CHECK-C: "x86_64-unknown-linux-gnu" - "clang"
// CHECK-C: "x86_64-unknown-linux-gnu" - "clang::as"
// CHECK-C: "x86_64-unknown-linux-gnu" - "offload bundler"

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -emit-llvm -S -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-EMIT-LLVM-IR
// CHECK-EMIT-LLVM-IR: clang{{.*}}"-cc1"{{.*}}"-triple" "amdgcn-amd-amdhsa"{{.*}}"-emit-llvm"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -lm --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIB-DEVICE
// CHECK-LIB-DEVICE: {{.*}}llvm-link{{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803 -lm --rocm-device-lib-path=%S/Inputs/rocm/amdgcn/bitcode -fopenmp-new-driver %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIB-DEVICE-NEW
// CHECK-LIB-DEVICE-NEW: {{.*}}clang-linker-wrapper{{.*}}-target-library=amdgcn-amd-amdhsa-gfx803={{.*}}ocml.bc"{{.*}}ockl.bc"{{.*}}oclc_daz_opt_on.bc"{{.*}}oclc_unsafe_math_off.bc"{{.*}}oclc_finite_only_off.bc"{{.*}}oclc_correctly_rounded_sqrt_on.bc"{{.*}}oclc_wavefrontsize64_on.bc"{{.*}}oclc_isa_version_803.bc"
