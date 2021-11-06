// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-mingw32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-windows-msvc-elf | FileCheck %s --check-prefix=ELF32
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-mingw32 | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc-elf | FileCheck %s --check-prefix=ELF64

// CHECK: target datalayout = "e-m:x-{{.*}}"
// X64: target datalayout = "e-m:w-{{.*}}"
// ELF32: target datalayout = "e-m:e-{{.*}}"
// ELF64: target datalayout = "e-m:e-{{.*}}"

void __stdcall f1(void) {}
// CHECK: define dso_local x86_stdcallcc void @"\01_f1@0"
// X64: define dso_local void @f1(
// ELF32: define{{.*}} x86_stdcallcc void @"\01_f1@0"
// ELF64: define{{.*}} void @f1(

void __fastcall f2(void) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f2@0"
// X64: define dso_local void @f2(
// ELF32: define{{.*}} x86_fastcallcc void @"\01@f2@0"
// ELF64: define{{.*}} void @f2(

void __stdcall f3() {}
// CHECK: define dso_local x86_stdcallcc void @"\01_f3@0"
// X64: define dso_local void @f3(

void __fastcall f4(char a) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f4@4"
// X64: define dso_local void @f4(

void __fastcall f5(short a) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f5@4"
// X64: define dso_local void @f5(

void __fastcall f6(int a) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f6@4"
// X64: define dso_local void @f6(

void __fastcall f7(long a) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f7@4"
// X64: define dso_local void @f7(

void __fastcall f8(long long a) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f8@8"
// X64: define dso_local void @f8(

void __fastcall f9(long long a, char b, char c, short d) {}
// CHECK: define dso_local x86_fastcallcc void @"\01@f9@20"(i64 noundef %a, i8 noundef signext %b, i8 noundef signext %c, i16 noundef signext %d)
// X64: define dso_local void @f9(

void f12(void) {}
// CHECK: define dso_local void @f12(
// X64: define dso_local void @f12(

void __vectorcall v1(void) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v1@@0"(
// X64: define dso_local x86_vectorcallcc void @"\01v1@@0"(
// ELF32: define{{.*}} x86_vectorcallcc void @"\01v1@@0"(
// ELF64: define{{.*}} x86_vectorcallcc void @"\01v1@@0"(

void __vectorcall v2(char a) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v2@@4"(
// X64: define dso_local x86_vectorcallcc void @"\01v2@@8"(
// ELF32: define{{.*}} x86_vectorcallcc void @"\01v2@@4"(
// ELF64: define{{.*}} x86_vectorcallcc void @"\01v2@@8"(

void __vectorcall v3(short a) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v3@@4"(
// X64: define dso_local x86_vectorcallcc void @"\01v3@@8"(

void __vectorcall v4(int a) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v4@@4"(
// X64: define dso_local x86_vectorcallcc void @"\01v4@@8"(

void __vectorcall v5(long long a) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v5@@8"(
// X64: define dso_local x86_vectorcallcc void @"\01v5@@8"(

void __vectorcall v6(char a, char b) {}
// CHECK: define dso_local x86_vectorcallcc void @"\01v6@@8"(
// X64: define dso_local x86_vectorcallcc void @"\01v6@@16"(
