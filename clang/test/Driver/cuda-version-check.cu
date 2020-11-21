// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_20 --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// Test version guess when no version.txt or cuda.h are found
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA-unknown/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=UNKNOWN_VERSION
// Unknown version with version.txt present
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=UNKNOWN_VERSION_V
// Unknown version with no version.txt but with version info present in cuda.h
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=UNKNOWN_VERSION_H
// Make sure that we don't warn about CUDA version during C++ compilation.
// RUN: %clang --target=x86_64-linux -v -### -x c++ --cuda-gpu-arch=sm_60 \
// RUN:    --cuda-path=%S/Inputs/CUDA-unknown/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=UNKNOWN_VERSION_CXX

// The installation at Inputs/CUDA is CUDA 7.0, which doesn't support sm_60.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// This should only complain about sm_60, not sm_35.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_35 \
// RUN:    --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60 --check-prefix=OK_SM35

// We should get two errors here, one for sm_60 and one for sm_61.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_61 \
// RUN:    --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60 --check-prefix=ERR_SM61

// We should still get an error if we pass -nocudainc, because this compilation
// would invoke ptxas, and we do a version check on that, too.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 -nocudainc --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// If with -nocudainc and -E, we don't touch the CUDA install, so we
// shouldn't get an error.
// RUN: %clang --target=x86_64-linux -v -### -E --cuda-device-only --cuda-gpu-arch=sm_60 -nocudainc \
// RUN:    --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK

// --no-cuda-version-check should suppress all of these errors.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 \
// RUN:    --no-cuda-version-check %s | \
// RUN:    FileCheck %s --check-prefix=OK

// We need to make sure the version check is done only for the device toolchain,
// therefore we should not get an error in host-only mode. We use the -S here
// to avoid the error being produced in case by the assembler tool, which does
// the same check.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-host-only --cuda-path=%S/Inputs/CUDA/usr/local/cuda -S 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-device-only --cuda-path=%S/Inputs/CUDA/usr/local/cuda -S 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// OK-NOT: error: GPU arch

// OK_SM35-NOT: error: GPU arch sm_35

// We should only get one error per architecture.
// ERR_SM60: error: GPU arch sm_60 {{.*}}
// ERR_SM60-NOT: error: GPU arch sm_60

// ERR_SM61: error: GPU arch sm_61 {{.*}}
// ERR_SM61-NOT: error: GPU arch sm_61

// UNKNOWN_VERSION_V: Unknown CUDA version. version.txt:{{.*}}. Assuming the latest supported version
// UNKNOWN_VERSION_H: Unknown CUDA version. cuda.h: CUDA_VERSION={{.*}}. Assuming the latest supported version
// UNKNOWN_VERSION: Unknown CUDA version. No version found in version.txt or cuda.h. Assuming the latest supported version
// UNKNOWN_VERSION_CXX-NOT: Unknown CUDA version
