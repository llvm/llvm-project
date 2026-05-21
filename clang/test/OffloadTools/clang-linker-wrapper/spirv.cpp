// Verify the ELF packaging of OpenMP SPIR-V device images.
// REQUIRES: system-linux
// REQUIRES: spirv-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -aux-triple spirv64-intel \
// RUN:   -emit-llvm-bc -emit-llvm-uselists -disable-llvm-passes -fopenmp \
// RUN:   --offload-new-driver --no-offloadlib --offload-targets=spirv64-intel \
// RUN:   -fcxx-exceptions -fexceptions -x c++ -o %t.bc
// RUN: %clang -cc1 %s -triple spirv64-intel -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -emit-obj -fopenmp --offload-new-driver --no-offloadlib \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.bc \
// RUN:   -fcxx-exceptions -fexceptions -x c++ -o %t.device.o
// RUN: llvm-offload-binary -o %t.bundle \
// RUN:   --image=file=%t.device.o,triple=spirv64-intel,arch=generic,kind=openmp
// RUN: %clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64-intel \
// RUN:   -emit-obj -fopenmp --offload-new-driver --no-offloadlib \
// RUN:   --offload-targets=spirv64-intel -fembed-offload-object=%t.bundle \
// RUN:   -x ir %t.bc -o %t.host.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --device-linker=spirv64-intel=--allow-partial-linkage \
// RUN:   --device-linker=spirv64-intel=--create-library \
// RUN:   --linker-path=/usr/bin/ld %t.host.o -o %t
// RUN: llvm-objdump --offloading %t | FileCheck -check-prefix=CHECK %s

// CHECK: nested images   1
// CHECK:   triple          spirv64-intel

// Stub symbols required by the OpenMP offload descriptor registration code
// emitted by clang-linker-wrapper; avoids linking -lomp/-lomptarget/-lc.
extern "C" {
void __tgt_register_lib(void *) {}
void __tgt_unregister_lib(void *) {}
int atexit(void (*)(void)) { return 0; }
}

int main(int argc, char** argv) {
  return 0;
}