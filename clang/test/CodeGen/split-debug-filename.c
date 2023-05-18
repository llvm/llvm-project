// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -debug-info-kind=limited -split-dwarf-file foo.dwo -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -split-dwarf-file %t.dwo -split-dwarf-output %t.dwo -emit-obj -o - %s | llvm-readobj -S - | FileCheck --check-prefix=O %s
// RUN: llvm-readobj -S %t.dwo | FileCheck --check-prefix=DWO %s

int main (void) {
  return 0;
}

// Testing to ensure that the dwo name gets output into the compile unit.
// CHECK: !DICompileUnit({{.*}}, splitDebugFilename: "foo.dwo"

// O-NOT: .dwo
// DWO: .dwo
