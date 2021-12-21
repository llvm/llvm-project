// REQUIRES: x86-registered-target
// RUN: cp %s %T/debug-info-objname.cpp
// RUN: cd %T

// No output file provided, input file is relative, we emit an absolute path (MSVC behavior).
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

// No output file provided, input file is absolute, we emit an absolute path (MSVC behavior).
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -- %T/debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

// The output file is provided as an absolute path, we emit an absolute path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc /Fo%T/debug-info-objname.obj -- %T/debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

// The output file is provided as relative path, -working-dir is provided, we emit an absolute path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -working-dir=%T debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

// The input file name is relative and we specify -fdebug-compilation-dir, we emit a relative path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=RELATIVE

// Ensure /FA emits an .asm file which contains the path to the final .obj, not the .asm
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. /FA debug-info-objname.cpp
// RUN: cat debug-info-objname.asm | FileCheck %s --check-prefix=ASM

// Same thing for -save-temps
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. /clang:-save-temps debug-info-objname.cpp
// RUN: cat debug-info-objname.asm | FileCheck %s --check-prefix=ASM

int main() {
  return 1;
}

// ABSOLUTE: S_OBJNAME [size = {{[0-9]+}}] sig=0, `{{.+}}debug-info-objname.obj`
// RELATIVE: S_OBJNAME [size = {{[0-9]+}}] sig=0, `debug-info-objname.obj`
// ASM: Record kind: S_OBJNAME
// ASM-NEXT: .long   0
// ASM-NEXT: .asciz  "debug-info-objname.obj"
