// REQUIRES: clang-driver

// Check that we raise an error if we're trying to compile OpenCL for amdhsa code but can't
// find a ROCm install, unless -nogpulib was passed.

// RUN: %clang -### --sysroot=%s/no-rocm-there -target amdgcn--amdhsa %s 2>&1 | FileCheck %s --check-prefix ERR
// RUN: %clang -### --rocm-path=%s/no-rocm-there -target amdgcn--amdhsa %s 2>&1 | FileCheck %s --check-prefix ERR
// ERR: cannot find ROCm installation. Provide its path via --rocm-path, or pass -nogpulib and -nogpuinc to build without ROCm device library and HIP includes.

// Accept nogpulib or nostdlib for OpenCL.
// RUN: %clang -### -nogpulib --rocm-path=%s/no-rocm-there %s 2>&1 | FileCheck %s --check-prefix OK
// RUN: %clang -### -nostdlib --rocm-path=%s/no-rocm-there %s 2>&1 | FileCheck %s --check-prefix OK
// OK-NOT: cannot find ROCm installation.
