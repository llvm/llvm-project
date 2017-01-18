// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_20 --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_20 --sysroot=%S/Inputs/CUDA_80 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --sysroot=%S/Inputs/CUDA_80 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK

// The installation at Inputs/CUDA is CUDA 7.0, which doesn't support sm_60.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// This should only complain about sm_60, not sm_35.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_35 \
// RUN:    --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60 --check-prefix=OK_SM35

// We should get two errors here, one for sm_60 and one for sm_61.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_61 \
// RUN:    --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60 --check-prefix=ERR_SM61

// We should still get an error if we pass -nocudainc, because this compilation
// would invoke ptxas, and we do a version check on that, too.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 -nocudainc --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// If with -nocudainc and -E, we don't touch the CUDA install, so we
// shouldn't get an error.
// RUN: %clang --target=x86_64-linux -v -### -E --cuda-device-only --cuda-gpu-arch=sm_60 -nocudainc \
// RUN:    --sysroot=%S/Inputs/CUDA 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK

// --no-cuda-version-check should suppress all of these errors.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --sysroot=%S/Inputs/CUDA 2>&1 \
// RUN:    --no-cuda-version-check %s | \
// RUN:    FileCheck %s --check-prefix=OK

// We need to make sure the version check is done only for the device toolchain,
// therefore we should not get an error in host-only mode. We use the -S here
// to avoid the error being produced in case by the assembler tool, which does
// the same check.
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-host-only --sysroot=%S/Inputs/CUDA -S 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=OK
// RUN: %clang --target=x86_64-linux -v -### --cuda-gpu-arch=sm_60 --cuda-device-only --sysroot=%S/Inputs/CUDA -S 2>&1 %s | \
// RUN:    FileCheck %s --check-prefix=ERR_SM60

// OK-NOT: error: GPU arch

// OK_SM35-NOT: error: GPU arch sm_35

// We should only get one error per architecture.
// ERR_SM60: error: GPU arch sm_60 {{.*}}
// ERR_SM60-NOT: error: GPU arch sm_60

// ERR_SM61: error: GPU arch sm_61 {{.*}}
// ERR_SM61-NOT: error: GPU arch sm_61
