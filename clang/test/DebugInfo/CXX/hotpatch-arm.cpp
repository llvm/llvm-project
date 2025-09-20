// REQUIRES: arm-registered-target
///
/// Check that using /hotpatch doesn't generate an error.
/// Binaries are always hotpatchable on ARM/ARM64.
///
// RUN: %clang_cl --target=arm-pc-windows-msvc /c /hotpatch /Z7 -- %s 2>&1
///
/// Ensure that we set the hotpatchable flag in the debug information.
///
// RUN: %clang_cl --target=arm-pc-windows-msvc /c /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// HOTPATCH: S_COMPILE3 [size = [[#]]]
// HOTPATCH: flags = hot patchable
//
// RUN: %clang_cl --target=arm-pc-windows-msvc /c -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH

int main() {
  return 0;
}
