// Checks that cuda compilation does the right thing when passed -fcuda-short-ptr

// RUN: %clang -### --target=x86_64-linux-gnu -c -march=haswell --cuda-gpu-arch=sm_20 -fcuda-short-ptr -nocudainc -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 | FileCheck %s

// CHECK: "-mllvm" "--nvptx-short-ptr"
// CHECK-SAME: "-fcuda-short-ptr"
