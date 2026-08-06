// Host-side compilation on x86 (no errors expected).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64 -fopenmp -x c -fsyntax-only -verify=host %s

// Device-side compilation for targets without 128-bit float/complex support (no errors expected).
// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -fsyntax-only -verify=device %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -fsyntax-only -verify=device %s
// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -fsyntax-only -verify=device %s

// host-no-diagnostics
// device-no-diagnostics
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
