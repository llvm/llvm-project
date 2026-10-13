// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-obj -gcodeview -debug-info-kind=limited -fms-hotpatch %s -o %t.obj
// RUN: llvm-readobj --codeview %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-obj -gcodeview -debug-info-kind=limited %s -o %t.obj
// RUN: llvm-readobj --codeview %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-obj -gcodeview -debug-info-kind=limited -fms-hotpatch %s -o %t.obj
// RUN: llvm-readobj --codeview %t.obj | FileCheck %s --check-prefix=HOTPATCH
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -emit-obj -gcodeview -debug-info-kind=limited %s -o %t.obj
// RUN: llvm-readobj --codeview %t.obj | FileCheck %s --check-prefix=NO-HOTPATCH

// HOTPATCH:      Compile3Sym {
// HOTPATCH:        Flags [ (0x4000)
// HOTPATCH-NEXT:     HotPatch (0x4000)
// HOTPATCH-NEXT:   ]

// NO-HOTPATCH:      Compile3Sym {
// NO-HOTPATCH:        Flags [ (0x0)
// NO-HOTPATCH-NEXT:   ]

void f(void) {}
