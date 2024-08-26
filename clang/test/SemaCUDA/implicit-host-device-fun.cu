// RUN: %clang_cc1 -isystem %S/Inputs  -fsyntax-only %s
// RUN: %clang_cc1 -isystem %S/Inputs -fcuda-is-device  -fsyntax-only %s
// RUN: %clang_cc1 -isystem %S/Inputs -foffload-implicit-host-device-templates -fsyntax-only %s
// RUN: %clang_cc1 -isystem %S/Inputs -foffload-implicit-host-device-templates -fcuda-is-device  -fsyntax-only %s

#include <cuda.h>

template<typename T>
void tempf(T x) {
}

template<typename T>
__device__ void tempf(T x) {
}

void host_fun() {
  tempf(1);
}

__device__ void device_fun() {
  tempf(1);
}
