// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-SEH
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-SJLJ
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep-1.7 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP-1_7
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNUSTEP
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SEH
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SJLJ
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SEH
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SJLJ

// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep-1.7 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC
// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-MSVC

// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-SEH
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx-fragile -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MACOSX-FRAGILE-SJLJ
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=macosx -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=ios -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=watchos -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-NS
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep-1.7 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-GNU
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gnustep -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-GNU
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SEH
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=gcc -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GCC-SJLJ
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SEH
// RUN: %clang_cc1 -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fobjc-exceptions -fcxx-exceptions -fobjc-runtime=objfw -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-OBJFW-SJLJ

void g(void);

@interface use
+ (void)e:(id)xception;
@end

// CHECK-MACOSX-FRAGILE: personality ptr @__gxx_personality_v0
// CHECK-MACOSX-FRAGILE-SEH: personality ptr @__gxx_personality_seh0
// CHECK-MACOSX-FRAGILE-SJLJ: personality ptr @__gxx_personality_sj0
// CHECK-NS: personality ptr @__objc_personality_v0
// CHECK-GNUSTEP-1_7: personality ptr @__gnustep_objcxx_personality_v0
// CHECK-GNUSTEP: personality ptr @__gnustep_objcxx_personality_v0
// CHECK-GCC: personality ptr @__gnu_objc_personality_v0
// CHECK-GCC-SEH: personality ptr @__gnu_objc_personality_seh0
// CHECK-GCC-SJLJ: personality ptr @__gnu_objc_personality_sj0
// CHECK-OBJFW: personality ptr @__gnu_objc_personality_v0
// CHECK-OBJFW-SEH: personality ptr @__gnu_objc_personality_seh0
// CHECK-OBJFW-SJLJ: personality ptr @__gnu_objc_personality_sj0

// CHECK-WIN-MSVC: personality ptr @__CxxFrameHandler3
// CHECK-WIN-GNU: personality ptr @__gxx_personality_seh0

void f(void) {
  @try {
    g();
  } @catch (id xception) {
    [use e:xception];
  }
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


