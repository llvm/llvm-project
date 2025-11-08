// REQUIRES: x86-registered-target
///
// RUN: %clang_cl --target=x86_64-windows-msvc /c /hotpatch /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// HOTPATCH: S_COMPILE3 [size = [[#]]]
// HOTPATCH: flags = hot patchable
///
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
// NO-HOTPATCH-NOT: flags = hot patchable
///
// RUN: %clang_cl --target=x86_64-windows-msvc /hotpatch -### -- %s 2>&1 \
// RUN:    | FileCheck %s --check-prefix=FUNCTIONPADMIN
// FUNCTIONPADMIN: clang{{.*}}
// FUNCTIONPADMIN: {{link[^"]*"}} 
// FUNCTIONPADMIN: -functionpadmin

int main() {
  return 0;
}
