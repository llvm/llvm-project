// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -S -o - %s \
// RUN:   | FileCheck --check-prefix=DEFAULT %s

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -fapprox-func -S -o - %s \
// RUN:   | FileCheck --check-prefix=APPROX %s

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -fapprox-func \
// RUN:   -mllvm -nvptx-prec-sqrtf32=1 -S -o - %s \
// RUN:   | FileCheck --check-prefix=FORCE-PRECISE %s

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -target-cpu sm_75 -O2 -mllvm -nvptx-prec-sqrtf32=0 -S -o - %s \
// RUN:   | FileCheck --check-prefix=FORCE-APPROX %s

// DEFAULT-LABEL: .func{{.*}}builtin_sqrtf
// DEFAULT: sqrt.rn.f32
// DEFAULT-NOT: sqrt.approx.f32
//
// APPROX-LABEL: .func{{.*}}builtin_sqrtf
// APPROX: sqrt.approx.f32
// APPROX-NOT: sqrt.rn.f32
//
// FORCE-PRECISE-LABEL: .func{{.*}}builtin_sqrtf
// FORCE-PRECISE: sqrt.rn.f32
// FORCE-PRECISE-NOT: sqrt.approx.f32
//
// FORCE-APPROX-LABEL: .func{{.*}}builtin_sqrtf
// FORCE-APPROX: sqrt.approx.f32
// FORCE-APPROX-NOT: sqrt.rn.f32
extern "C" __attribute__((device)) float builtin_sqrtf(float x) {
  return __builtin_sqrtf(x);
}
