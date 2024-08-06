// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

// Check that -gheterogeneous-dwarf without an `=OPTION` suffix remains valid
// and aliases the new default. This is needed for transitioning flang-legacy
// as it depends on the -cc1 interface.

// CHECK: #dbg_declare{{.*}}DIExpression{{.*}}DIOp
__attribute__((device)) void kernel1(int Arg) {
  int FuncVar;
}
