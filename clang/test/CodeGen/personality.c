// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DWARF
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SEH
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SJLJ

// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X86
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X64

// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DWARF
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SEH
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fblocks -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-SJLJ


extern void g(void (^)(void));
extern void i(void);

// CHECK-GNU: personality ptr @__gcc_personality_v0
// CHECK-DWARF: personality ptr @__gcc_personality_v0
// CHECK-SEH: personality ptr @__gcc_personality_seh0
// CHECK-SJLJ: personality ptr @__gcc_personality_sj0

// CHECK-WIN: personality ptr @__CxxFrameHandler3
// CHECK-MINGW-SEH: personality ptr @__gcc_personality_seh0

void f(void) {
  __block int i;
  ^{ (void)i; };
  g(^ { });
}

#if defined(__SEH_EXCEPTIONS__)
// CHECK-WIN-SEH-X86: personality ptr @_except_handler3
// CHECK-WIN-SEH-X64: personality ptr @__C_specific_handler

void h(void) {
  __try {
    i();
  } __finally {
  }
}
#endif

