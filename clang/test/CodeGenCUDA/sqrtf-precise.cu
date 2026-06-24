// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -S -o - %s \
// RUN:   -include __clang_cuda_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/../Headers/Inputs/include \
// RUN:   | FileCheck --check-prefix=RN %s

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -fapprox-func -S -o - %s \
// RUN:   -include __clang_cuda_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/../Headers/Inputs/include \
// RUN:   | FileCheck --check-prefix=APPROX %s

#include <math.h>

// RN-LABEL: .func{{.*}}_Z1ff
// RN: sqrt.rn.f32
// RN-NOT: sqrt.approx.f32
//
// APPROX-LABEL: .func{{.*}}_Z1ff
// APPROX: sqrt.approx.f32
__device__ float f(float x) {
  return sqrtf(x);
}
