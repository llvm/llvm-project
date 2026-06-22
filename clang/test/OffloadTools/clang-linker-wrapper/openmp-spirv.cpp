// Verify the ELF packaging of OpenMP SPIR-V device images.
// REQUIRES: system-linux
// REQUIRES: spirv-tools
// REQUIRES: spirv-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.bc
// RUN: %clang_cc1 %s -triple spirv64-intel -fopenmp-is-target-device -emit-obj -o %t.device.o
// RUN: llvm-offload-binary -o %t.bundle --image=file=%t.device.o,triple=spirv64-intel,arch=generic,kind=openmp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -fembed-offload-object=%t.bundle -x ir %t.bc -o %t.host.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld %t.host.o -o %t
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
