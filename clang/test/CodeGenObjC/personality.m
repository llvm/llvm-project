// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=ios -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=macosx -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=watchos -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-1.7 -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP-1_7
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SJLJ
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SJLJ

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=ios -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=macosx -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=watchos -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-1.7 -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-MINGW-DWARF
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-MINGW-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fobjc-runtime=macosx-fragile -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-MINGW-SJLJ
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=ios -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=macosx -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=watchos -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-1.7 -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP-1_7
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fobjc-runtime=gcc -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SJLJ
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fobjc-runtime=objfw -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SJLJ

void g(void);

// CHECK-MACOSX-FRAGILE: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
// CHECK-NS: personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
// CHECK-GNUSTEP-1_7: personality i8* bitcast (i32 (...)* @__gnustep_objc_personality_v0 to i8*)
// CHECK-GNUSTEP: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_v0 to i8*)
// CHECK-GCC: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_v0 to i8*)
// CHECK-GCC-SEH: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_seh0 to i8*)
// CHECK-GCC-SJLJ: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_sj0 to i8*)
// CHECK-OBJFW: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_v0 to i8*)
// CHECK-OBJFW-SEH: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_seh0 to i8*)
// CHECK-OBJFW-SJLJ: personality i8* bitcast (i32 (...)* @__gnu_objc_personality_sj0 to i8*)

// CHECK-WIN-MSVC: personality i8* bitcast (i32 (...)* @__CxxFrameHandler3  to i8*)

// CHECK-MACOSX-FRAGILE-MINGW-DWARF: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
// CHECK-MACOSX-FRAGILE-MINGW-SEH: personality i8* bitcast (i32 (...)* @__gcc_personality_seh0 to i8*)
// CHECK-MACOSX-FRAGILE-MINGW-SJLJ: personality i8* bitcast (i32 (...)* @__gcc_personality_sj0 to i8*)

void f(void) {
  @try {
    g();
  } @catch (...) {
  }
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

