// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-OGCG --input-file=%t.ll %s

// Test wincall calling convention in CIR
// WinCall is the default for x86_64apx Windows targets or via the wincall attribute

void __attribute__((wincall)) wc(int a, int b, int c);
void __attribute__((wincall)) caller(void) { wc(1, 2, 3); }
void __attribute__((wincall)) plain(int a, int b, int c) { wc(a, b, c); }

// CIR: cir.func no_inline dso_local @caller{{.*}}cc(x86_wincall)
// CIR: cir.func private @wc{{.*}}cc(x86_wincall)
// CIR: cir.func no_inline dso_local @plain{{.*}}cc(x86_wincall)

// LLVM-CIR: define dso_local x86_wincallcc void @caller()
// LLVM-CIR: declare x86_wincallcc void @wc(i32 noundef, i32 noundef, i32 noundef)
// LLVM-CIR: define dso_local x86_wincallcc void @plain(i32 noundef %0, i32 noundef %1, i32 noundef %2)

// LLVM-OGCG: define dso_local x86_wincallcc void @caller()
// LLVM-OGCG: declare x86_wincallcc void @wc(i32 noundef, i32 noundef, i32 noundef)
// LLVM-OGCG: define dso_local x86_wincallcc void @plain(i32 noundef %a, i32 noundef %b, i32 noundef %c)
