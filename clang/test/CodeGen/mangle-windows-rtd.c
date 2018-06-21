// RUN: %clang_cc1 -emit-llvm -mrtd %s -o - -triple=i386-mingw32 | FileCheck %s

void f1(void) {}
// CHECK: define x86_stdcallcc void @"\01_f1@0"

void __stdcall f2(void) {}
// CHECK: define x86_stdcallcc void @"\01_f2@0"

void __fastcall f3(void) {}
// CHECK: define x86_fastcallcc void @"\01@f3@0"
