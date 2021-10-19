// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s --check-prefix=DARWIN
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fasm-blocks -emit-llvm -o - | FileCheck %s --check-prefix=WINDOWS

// On Windows, .align is in bytes, and on Darwin, .align is in log2 form. The
// Intel inline assembly parser should rewrite to the appropriate form depending
// on the platform.

void align_test() {
  __asm align 8
  __asm align 16;
  __asm align 128;
  __asm ALIGN 256;
}

// DARWIN-LABEL: define{{.*}} void @align_test()
// DARWIN: call void asm sideeffect inteldialect
// DARWIN-SAME: .align 3
// DARWIN-SAME: .align 4
// DARWIN-SAME: .align 7
// DARWIN-SAME: .align 8
// DARWIN-SAME: "~{dirflag},~{fpsr},~{flags}"()

// WINDOWS-LABEL: define dso_local void @align_test()
// WINDOWS: call void asm sideeffect inteldialect
// WINDOWS-SAME: .align 8
// WINDOWS-SAME: .align 16
// WINDOWS-SAME: .align 128
// WINDOWS-SAME: .align 256
// WINDOWS-SAME: "~{dirflag},~{fpsr},~{flags}"()
