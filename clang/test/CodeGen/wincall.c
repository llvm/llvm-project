// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-msvc -emit-llvm  -o - %s | FileCheck -check-prefix=MSVC %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-gnu -emit-llvm  -o - %s | FileCheck -check-prefix=GNU %s
// RUN: %clang_cc1 -triple x86_64apx-pc-windows-cygnus -emit-llvm  -o - %s | FileCheck -check-prefix=GNU %s
// RUN: %clang_cc1 -triple x86_64apx-pc-windows-msys -emit-llvm  -o - %s | FileCheck -check-prefix=GNU %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-uefi -emit-llvm  -o - %s | FileCheck -check-prefix=GNU %s

// The wincall calling convention is the default for x86_64apx PE/COFF
// targets (Windows, Cygwin, MSYS and UEFI) and appends a @win suffix to the
// symbol so the linker can catch calling convention mismatches.

void plain(int, int, int);

void __attribute__((wincall)) wc(int, int, int);

void caller(void) {
  // MSVC-LABEL: define dso_local x86_wincallcc void @"\01caller@win"
  // GNU-LABEL: define dso_local x86_wincallcc void @"\01caller@win"
  plain(1, 2, 3);
  // MSVC: call x86_wincallcc void @"\01plain@win"(i32 noundef 1, i32 noundef 2, i32 noundef 3)
  // GNU: call x86_wincallcc void @"\01plain@win"(i32 noundef 1, i32 noundef 2, i32 noundef 3)
  wc(1, 2, 3);
  // MSVC: call x86_wincallcc void @"\01wc@win"(i32 noundef 1, i32 noundef 2, i32 noundef 3)
  // GNU: call x86_wincallcc void @"\01wc@win"(i32 noundef 1, i32 noundef 2, i32 noundef 3)
}

void plain(int a, int b, int c) {
  // MSVC-LABEL: define dso_local x86_wincallcc void @"\01plain@win"
  // GNU-LABEL: define dso_local x86_wincallcc void @"\01plain@win"
  wc(a, b, c);
  // MSVC: call x86_wincallcc void @"\01wc@win"
  // GNU: call x86_wincallcc void @"\01wc@win"
}

void __attribute__((wincall)) wc(int a, int b, int c) {
  // MSVC-LABEL: define dso_local x86_wincallcc void @"\01wc@win"
  // GNU-LABEL: define dso_local x86_wincallcc void @"\01wc@win"
}
