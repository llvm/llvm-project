// Test that RISC-V host target features are correctly passed to the linker wrapper compilation
// and applied to wrapper object generation.
// UNSUPPORTED: system-windows
// REQUIRES: riscv-registered-target
// REQUIRES: nvptx-registered-target

// Simple program that requires OpenMP offloading
int main() {
#pragma omp target
    {
        // Device code
    }
    return 0;
}

// Verify that the driver invokes clang-linker-wrapper with correct host features.

// Test lp64d (double-float) ABI
// RUN: %clang -target riscv64-linux-gnu -mabi=lp64d \
// RUN:   -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
// RUN:   --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc \
// RUN:   %s -o %t.lp64d -### 2>&1 | FileCheck %s --check-prefix=LP64D-DRIVER

// Check that the driver calls clang-linker-wrapper with correct host features for lp64d
// LP64D-DRIVER: clang-linker-wrapper{{.*}}--host-triple=riscv64{{.*}}--host-features={{.*}}+f{{.*}}+d{{.*}}

// Test lp64f (single-float) ABI  
// RUN: %clang -target riscv64-linux-gnu -mabi=lp64f \
// RUN:   -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
// RUN:   --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc \
// RUN:   %s -o %t.lp64f -### 2>&1 | FileCheck %s --check-prefix=LP64F-DRIVER

// LP64F-DRIVER: clang-linker-wrapper{{.*}}--host-triple=riscv64{{.*}}--host-features={{.*}}+f{{.*}}-d{{.*}}

// Test lp64 (soft-float) ABI
// RUN: %clang -target riscv64-linux-gnu -mabi=lp64 \
// RUN:   -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
// RUN:   --libomptarget-nvptx-bc-path=%S/Inputs/libomptarget/libomptarget-nvptx-test.bc \
// RUN:   %s -o %t.lp64 -### 2>&1 | FileCheck %s --check-prefix=LP64-DRIVER

// LP64-DRIVER: clang-linker-wrapper{{.*}}--host-triple=riscv64{{.*}}--host-features={{.*}}-f{{.*}}-d{{.*}}

// Verify that clang-linker-wrapper applies RISC-V host features correctly when creating wrapper objects.
// We do this by checking the ELF ABI flags in the generated wrapper object files.

// Create test objects for linker-wrapper testing
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.device.bc  
// RUN: llvm-offload-binary -o %t.openmp.out \
// RUN:   --image=file=%t.device.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple riscv64-unknown-linux-gnu -emit-obj -o %t.riscv.host.o -fembed-offload-object=%t.openmp.out

// Test lp64 (soft-float) ABI - should generate ELF flags 0x0
// RUN: rm -rf %t.tmpdir1 && mkdir %t.tmpdir1
// RUN: cd %t.tmpdir1 && clang-linker-wrapper --host-triple=riscv64-unknown-linux-gnu \
// RUN:   --host-features=-f,-d --linker-path=/usr/bin/ld %t.riscv.host.o -o %t.lp64.out --dry-run --save-temps 2>&1
// RUN: cd %t.tmpdir1 && find . -name "*.wrapper*.o" -exec llvm-readobj --file-headers "{}" ";" \
// RUN:   | FileCheck %s --check-prefix=SOFT-FLOAT-OBJ

// SOFT-FLOAT-OBJ: Flags [ (0x0)

// Test lp64f (single-float) ABI - should generate ELF flags 0x2
// RUN: rm -rf %t.tmpdir2 && mkdir %t.tmpdir2
// RUN: cd %t.tmpdir2 && clang-linker-wrapper --host-triple=riscv64-unknown-linux-gnu \
// RUN:   --host-features=+f,-d --linker-path=/usr/bin/ld %t.riscv.host.o -o %t.lp64f.out --dry-run --save-temps 2>&1
// RUN: cd %t.tmpdir2 && find . -name "*.wrapper*.o" -exec llvm-readobj --file-headers "{}" ";" \
// RUN:   | FileCheck %s --check-prefix=SINGLE-FLOAT-OBJ

// SINGLE-FLOAT-OBJ: Flags [ (0x2)  

// Test lp64d (double-float) ABI - should generate ELF flags 0x4
// RUN: rm -rf %t.tmpdir3 && mkdir %t.tmpdir3
// RUN: cd %t.tmpdir3 && clang-linker-wrapper --host-triple=riscv64-unknown-linux-gnu \
// RUN:   --host-features=+f,+d --linker-path=/usr/bin/ld %t.riscv.host.o -o %t.lp64d.out --dry-run --save-temps 2>&1
// RUN: cd %t.tmpdir3 && find . -name "*.wrapper*.o" -exec llvm-readobj --file-headers "{}" ";" \
// RUN:   | FileCheck %s --check-prefix=DOUBLE-FLOAT-OBJ

// DOUBLE-FLOAT-OBJ: Flags [ (0x4)
