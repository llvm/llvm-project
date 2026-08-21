// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LINUX-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LINUX-OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fclangir -emit-llvm %s -o %t-darwin-cir.ll
// RUN: FileCheck --check-prefix=DARWIN --input-file=%t-darwin-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o %t-darwin.ll
// RUN: FileCheck --check-prefix=DARWIN --input-file=%t-darwin.ll %s

// The 0.98 ABI revision sends an eightbyte pair to memory when the high half is
// X87UP and the low half is not X87.  Darwin exempts itself for binary
// compatibility with older GCC, so the same union passes in registers there.
// The int member is what makes the low half INTEGER rather than X87.
typedef union { long double l; int i; } ULongDouble;
void rev98(ULongDouble u) { (void)u; }

// LINUX-CIR: define dso_local void @rev98(ptr noalias noundef byval(%union.ULongDouble) align 16 %{{[^,)]+}})
// LINUX-OGCG: define dso_local void @rev98(ptr noundef byval(%union.ULongDouble) align 16 %{{[^,)]+}})
// DARWIN: define void @rev98(i64 %{{[^,)]+}}, double %{{[^,)]+}})
