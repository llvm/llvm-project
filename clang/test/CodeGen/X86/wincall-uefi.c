// RUN: %clang_cc1 -triple x86_64apx-unknown-uefi -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-uefi -O2 -mframe-pointer=all -o - -S %s | FileCheck -check-prefix=FRAME %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-uefi -o - -emit-llvm %s | FileCheck -check-prefix=SEH %s

// UEFI is a PE/COFF target, so x86_64apx-unknown-uefi also defaults to the
// wincall calling convention (8 integer argument registers incl. R16-R19,
// @win symbol suffix), and because UEFI uses WinCFI it needs unwind v3 so the
// push2/pop2 frame (pushp/popp) can describe its epilogues.

int add6(int a, int b, int c, int d, int e, int f) { return a + b + c + d + e + f; }
// CHECK-LABEL: define dso_local x86_wincallcc i32 @"\01add6@win"(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e, i32 noundef %f)

// The wincall default is also the ABI used for calls.
void callee(int, int, int);
void caller(void) {
  callee(1, 2, 3);
  // CHECK-LABEL: define dso_local x86_wincallcc void @"\01caller@win"
  // CHECK: call x86_wincallcc void @"\01callee@win"(i32 noundef 1, i32 noundef 2, i32 noundef 3)
}

// With a frame pointer the push2/pop2 frame is used for rbp; the epilogue
// restore needs unwind v3 (module-wide on UEFI, like Windows).
int framed(int a) {
  volatile int x = a;
  // FRAME-LABEL: framed@win:
  // FRAME: pushp %rbp
  // FRAME: popp %rbp
  return x;
}
// SEH: !llvm.module.flags = !{!0}
// SEH: !0 = !{i32 2, !"winx64-eh-unwind", i32 3}
