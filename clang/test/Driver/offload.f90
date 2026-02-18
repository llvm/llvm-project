// RUN: %clang -### --driver-mode=flang --target=x86_64-unknown-linux-gnu \
// RUN:   -resource-dir %S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   --rocm-path=%S/Inputs/rocm -fopenmp --offload-arch=gfx908 \
// RUN: %s 2>&1 | FileCheck %s --check-prefix=CHECK-FLANG-RT
// CHECK-FLANG-RT: clang-linker-wrapper{{.*}}"--device-linker=amdgcn-amd-amdhsa=-lflang_rt.runtime"{{.*}}"-lflang_rt.runtime"
