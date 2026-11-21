// NOTE: Do not autogenerate. -verify checks target-specific diagnostics.
// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -fcuda-is-device -x hip -DNO_SLEEP_FOREVER -fsyntax-only -verify=range,forever %s
// RUN: %clang_cc1 -triple amdgpu12.50s-amd-amdhsa -fcuda-is-device -x hip -DNO_SLEEP_FOREVER -fsyntax-only -verify=range,forever %s
// RUN: %clang_cc1 -triple amdgpu12.51-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=range,negative %s
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -target-feature -no-sleep-forever -fcuda-is-device -x hip -fsyntax-only -verify=range,negative %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgpu12.50-amd-amdhsa -x hip -DNO_SLEEP_FOREVER -fsyntax-only -verify=range %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgpu13-amd-amdhsa -x hip -DNO_SLEEP_FOREVER -fsyntax-only -verify=range %s
// RUN: %clang_cc1 -triple amdgpu12.5-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=range,negative %s
// RUN: %clang_cc1 -triple amdgpu13-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=range,negative %s
// RUN: %clang_cc1 -triple amdgpu13.10-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=range,negative %s

#include "Inputs/cuda.h"

__device__ void monitor_sleep_range() {
  __builtin_amdgcn_s_monitor_sleep(0);
  __builtin_amdgcn_s_monitor_sleep(127);
  __builtin_amdgcn_s_monitor_sleep(128); // range-error {{argument value 128 is outside the valid range [0, 127]}}
  __builtin_amdgcn_s_monitor_sleep(200); // range-error {{argument value 200 is outside the valid range [0, 127]}}
  __builtin_amdgcn_s_monitor_sleep(32767); // range-error {{argument value 32767 is outside the valid range [0, 127]}}
#ifdef NO_SLEEP_FOREVER
  __builtin_amdgcn_s_monitor_sleep(-32768); // forever-error {{argument value -32768 is outside the valid range [0, 127]}}
#else
  __builtin_amdgcn_s_monitor_sleep(-32768);
  __builtin_amdgcn_s_monitor_sleep(-32641);
  __builtin_amdgcn_s_monitor_sleep(-32640); // negative-error {{argument value -32640 is outside the valid range [-32768, -32641]}}
  __builtin_amdgcn_s_monitor_sleep(-1); // negative-error {{argument value -1 is outside the valid range [-32768, -32641]}}
#endif
}

template <short Duration> __device__ void monitor_sleep_template() {
  __builtin_amdgcn_s_monitor_sleep(Duration);
}
template __device__ void monitor_sleep_template<127>();
