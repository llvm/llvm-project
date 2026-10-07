// REQUIRES: aarch64-registered-target
//
// ARM64 instructions are atomically patchable, but /hotpatch is needed to
// ensure that the first instruction of a function is not a branch target.
// The CodeView hotpatchable flag must match this protection, with or without
// debug information.
//
// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /clang:-O1 /hotpatch /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: llvm-objdump -d --no-show-raw-insn %t.obj | FileCheck %s --check-prefix=HOTPATCH-CODE

// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /clang:-O1 /hotpatch -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: llvm-objdump -d --no-show-raw-insn %t.obj | FileCheck %s --check-prefix=HOTPATCH-CODE

// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /clang:-O1 /Z7 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
// RUN: llvm-objdump -d --no-show-raw-insn %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH-CODE

// RUN: %clang_cl --target=aarch64-pc-windows-msvc /c /clang:-O1 -o %t.obj -- %s
// RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
// RUN: llvm-objdump -d --no-show-raw-insn %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH-CODE
//
// HOTPATCH: S_COMPILE3 [size = [[#]]]
// HOTPATCH: flags = hot patchable
// NO-HOTPATCH: S_COMPILE3 [size = [[#]]]
// NO-HOTPATCH: flags = none
//
// HOTPATCH-CODE-LABEL: <loop>:
// HOTPATCH-CODE-NEXT:  0: nop
// HOTPATCH-CODE-NEXT:  4: ldr w8, [x0]
// HOTPATCH-CODE:       b.lo 0x4 <loop+0x4>
// NO-HOTPATCH-CODE-LABEL: <loop>:
// NO-HOTPATCH-CODE-NEXT:  0: ldr w8, [x0]
// NO-HOTPATCH-CODE:       b.lo 0x0 <loop>

extern "C" void loop(int *a, int *b) {
  do {
    ++*a++;
  } while (a < b);
}
