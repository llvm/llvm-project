// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x c -DNO_SLEEP_FOREVER -fsyntax-only -verify=range,forever %s
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -x cl -DNO_SLEEP_FOREVER -fsyntax-only -verify=range,forever %s
// RUN: %clang_cc1 -triple amdgpu13-amd-amdhsa -x c -fsyntax-only -verify=range,negative %s
// RUN: %clang_cc1 -triple amdgpu13-amd-amdhsa -x cl -fsyntax-only -verify=range,negative %s

void monitor_sleep_range(void) {
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
