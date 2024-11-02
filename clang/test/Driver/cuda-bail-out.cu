// Test clang driver bails out after one error during CUDA compilation.

#ifdef FORCE_ERROR
#error compilation failed
#endif

// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DFORCE_ERROR %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DFORCE_ERROR --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_60 \
// RUN:   %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DFORCE_ERROR --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_60 \
// RUN:   --cuda-device-only %s 2>&1 | FileCheck %s

#if defined(ERROR_HOST) && !defined(__CUDA_ARCH__)
#error compilation failed
#endif

#if defined(ERROR_SM35) && (__CUDA_ARCH__ == 350)
#error compilation failed
#endif

#if defined(ERROR_SM60) && (__CUDA_ARCH__ == 600)
#error compilation failed
#endif

// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_HOST --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_60 \
// RUN:   %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_SM35 --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_60 \
// RUN:   --cuda-device-only %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_SM60 --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_60 \
// RUN:   --cuda-device-only %s 2>&1 | FileCheck %s

// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_HOST -DERROR_SM35 --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-gpu-arch=sm_60 %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_HOST -DERROR_SM60 --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-gpu-arch=sm_60 %s 2>&1 | FileCheck %s
// RUN: not %clang -target powerpc64le-ibm-linux-gnu -fsyntax-only -nocudalib \
// RUN:   -nocudainc -DERROR_SM35 -DERROR_SM60 --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-gpu-arch=sm_60 %s 2>&1 | FileCheck %s


// CHECK: error: compilation failed
// CHECK-NOT: error: compilation failed
