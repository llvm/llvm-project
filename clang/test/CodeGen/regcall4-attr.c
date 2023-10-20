// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=i386-pc-win32       | FileCheck %s --check-prefixes=X86
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=x86_64-pc-win32     | FileCheck %s --check-prefixes=X64
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=i386-pc-linux-gnu   | FileCheck %s --check-prefixes=X86
// RUN: %clang_cc1 -emit-llvm %s -o - -ffreestanding -triple=x86_64-pc-linux-gnu | FileCheck %s --check-prefixes=X64

#include <xmmintrin.h>


void __attribute__((regcall4)) v1b(int a, int b) {}
// X86: define dso_local x86_regcallcc void @__regcall4__v1b(i32 inreg noundef %a, i32 inreg noundef %b) #[[#ATTR:]]
// X64: define dso_local x86_regcallcc void @__regcall4__v1b(i32 noundef %a, i32 noundef %b) #[[#ATTR:]]

// X86: attributes #[[#ATTR:]] = { {{.*}}regcall4{{.*}} }