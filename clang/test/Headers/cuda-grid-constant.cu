// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -include __clang_cuda_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include-grid-constant \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu sm_70 -target-feature +ptx77 -O2 -S \
// RUN:   -fcuda-is-device -o - %s \
// RUN:   -D __CLANG_GPU_DISABLE_MATH_WRAPPERS \
// RUN:   | FileCheck %s \
// RUN:   --implicit-check-not="mov.b64 {{%rd[0-9]+}}, kernel_param_1"

struct S {
  int x, y;
};

extern "C" __attribute__((global)) void kernel(
    __attribute__((grid_constant)) const S gc, const S plain, int *out) {
  out[0] = __isGridConstant(&gc);
  out[1] = __isGridConstant(&plain);
}

// CHECK-LABEL: kernel(
// The grid-constant byval parameter stays in the parameter address space.
// CHECK-DAG: mov.b64 %rd{{[0-9]+}}, kernel_param_0
// CHECK-DAG: cvta.param
// The plain byval parameter must be copied to local memory. If it is left in
// the parameter address space, isspacep.param would incorrectly return 1.
// CHECK-DAG: ld.param.b{{32|64}} {{.*}}, [kernel_param_1
// CHECK-DAG: st.local.b{{32|64}}
// CHECK-COUNT-2: isspacep.param
