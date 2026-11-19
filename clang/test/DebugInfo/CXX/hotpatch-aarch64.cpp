// REQUIRES: aarch64-registered-target
//
// Check that the CodeView hotpatchable flag reflects whether /hotpatch was
// used, with or without debug information. The code generated for /hotpatch is
// tested in llvm/test/CodeGen/AArch64/patchable-prologue.ll.
//
// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /hotpatch /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /hotpatch -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
//
// HOTPATCH: S_COMPILE3 [size = [[#]]]
// HOTPATCH: flags = hot patchable
// NO-HOTPATCH: S_COMPILE3 [size = [[#]]]
// NO-HOTPATCH: flags = none

int main() {
  return 0;
}
