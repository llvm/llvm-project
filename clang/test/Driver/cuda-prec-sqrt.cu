// Checks that the -fcuda-prec-sqrt flag is passed to the cc1 frontend.

// RUN: %clang -### --target=x86_64-linux-gnu -c -fcuda-prec-sqrt -nocudainc -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 | FileCheck %s

// CHECK: "-triple" "nvptx64-nvidia-cuda"
// CHECK-SAME: "-fcuda-prec-sqrt"
