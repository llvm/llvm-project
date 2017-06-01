// RUN: %clang --cuda-host-only -nocudainc -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null | FileCheck --check-prefix HOST %s
// RUN: %clang --cuda-device-only -nocudainc -nocudalib -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null | FileCheck --check-prefix DEVICE-NOFAST %s
// RUN: %clang -fcuda-approx-transcendentals --cuda-device-only -nocudainc -nocudalib -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null | FileCheck --check-prefix DEVICE-FAST %s
// RUN: %clang -ffast-math --cuda-device-only -nocudainc -nocudalib -target i386-unknown-linux-gnu -x cuda -E -dM -o - /dev/null | FileCheck --check-prefix DEVICE-FAST %s

// HOST-NOT: __CLANG_CUDA_APPROX_TRANSCENDENTALS__
// DEVICE-NOFAST-NOT: __CLANG_CUDA_APPROX_TRANSCENDENTALS__
// DEVICE-FAST: __CLANG_CUDA_APPROX_TRANSCENDENTALS__
