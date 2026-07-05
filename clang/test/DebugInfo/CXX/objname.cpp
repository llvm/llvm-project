// REQUIRES: x86-registered-target
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: cp %s debug-info-objname.cpp

/// No output file provided, input file is relative, we emit an absolute path (MSVC behavior).
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

/// No output file provided, input file is absolute, we emit an absolute path (MSVC behavior).
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -- %t/debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

/// The output file is provided as an absolute path, we emit an absolute path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc /Fo%t/debug-info-objname.obj -- %t/debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

/// The output file is provided as relative path, -working-dir is provided, we emit an absolute path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -working-dir=%t debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=ABSOLUTE

/// The input file name is relative and we specify -fdebug-compilation-dir, we emit a relative path.
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. debug-info-objname.cpp
// RUN: llvm-pdbutil dump -all debug-info-objname.obj | FileCheck %s --check-prefix=RELATIVE

/// Ensure /FA emits an .asm file which contains the path to the final .obj, not the .asm
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. /FA debug-info-objname.cpp
// RUN: FileCheck --input-file=debug-info-objname.asm --check-prefix=ASM %s

/// Same thing for -save-temps
// RUN: %clang_cl --target=x86_64-windows-msvc /c /Z7 -nostdinc -fdebug-compilation-dir=. /clang:-save-temps debug-info-objname.cpp
// RUN: FileCheck --input-file=debug-info-objname.asm --check-prefix=ASM %s

int main() {
  return 1;
}

// ABSOLUTE: S_OBJNAME [size = [[#]]] sig=0, `{{.+}}debug-info-objname.obj`
// RELATIVE: S_OBJNAME [size = [[#]]] sig=0, `debug-info-objname.obj`
// ASM: Record kind: S_OBJNAME
// ASM-NEXT: .long   0
// ASM-NEXT: .asciz  "debug-info-objname.obj"
