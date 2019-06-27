// RUN: %clang_cc1 -w -O2 -fblocks -triple i386-pc-linux-gnu -target-cpu pentium4 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clang_cc1 -w -O2 -fblocks -triple i386-netbsd -target-cpu pentium4 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NETBSD
// RUN: %clang_cc1 -w -O2 -fblocks -triple i386-apple-darwin9 -target-cpu yonah -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -w -O2 -fblocks -triple i386-pc-elfiamcu -mfloat-abi soft -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,IAMCU
// RUN: %clang_cc1 -w -O2 -fblocks -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN32

#include <mmintrin.h>
__m64 m64;
void callee(__m64 __m1, __m64 __m2);
__m64 caller(__m64 __m1, __m64 __m2)
{
// LINUX-LABEL: define x86_mmx @caller(x86_mmx %__m1.coerce, x86_mmx %__m2.coerce)
// LINUX: tail call void @callee(x86_mmx %__m2.coerce, x86_mmx %__m1.coerce)
// LINUX: ret x86_mmx
// NETBSD-LABEL: define x86_mmx @caller(x86_mmx %__m1.coerce, x86_mmx %__m2.coerce)
// NETBSD: tail call void @callee(x86_mmx %__m2.coerce, x86_mmx %__m1.coerce)
// NETBSD: ret x86_mmx
// DARWIN-LABEL: define i64 @caller(i64 %__m1.coerce, i64 %__m2.coerce)
// DARWIN: tail call void @callee(i64 %__m2.coerce, i64 %__m1.coerce)
// DARWIN: ret i64
// IAMCU-LABEL: define <1 x i64> @caller(i64 %__m1.coerce, i64 %__m2.coerce)
// IAMCU: tail call void @callee(i64 %__m2.coerce, i64 %__m1.coerce)
// IAMCU: ret <1 x i64>
// WIN32-LABEL: define dso_local <1 x i64> @caller(i64 %__m1.coerce, i64 %__m2.coerce)
// WIN32: call void @callee(i64 %__m2.coerce, i64 %__m1.coerce)
// WIN32: ret <1 x i64>
  callee(__m2, __m1);
  return m64;
}
