// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fdwarf-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DWARF
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fseh-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SEH
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fsjlj-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SJLJ

// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X86
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X64

// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fdwarf-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DWARF
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fseh-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SEH
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fsjlj-exceptions -fblocks -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SJLJ


extern void g(void (^)(void));
extern void i(void);

// CHECK-GNU: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
// CHECK-DWARF: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
// CHECK-SEH: personality i8* bitcast (i32 (...)* @__gcc_personality_seh0 to i8*)
// CHECK-SJLJ: personality i8* bitcast (i32 (...)* @__gcc_personality_sj0 to i8*)

// CHECK-WIN: personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CHECK-MINGW-SEH: personality i8* bitcast (i32 (...)* @__gcc_personality_seh0 to i8*)

void f(void) {
  __block int i;
  g(^ { });
}

#if defined(__SEH_EXCEPTIONS__)
// CHECK-WIN-SEH-X86: personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK-WIN-SEH-X64: personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)

void h(void) {
  __try {
    i();
  } __finally {
  }
}
#endif

