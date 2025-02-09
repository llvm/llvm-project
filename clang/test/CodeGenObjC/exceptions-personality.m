// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu  -emit-llvm -fobjc-runtime=gnustep-2.0 -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck --check-prefixes=CHECK-MINGW-OBJC2 < %t %s

// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu  -emit-llvm -fobjc-runtime=gcc -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck --check-prefixes=CHECK-MINGW-GCC < %t %s

// RUN: %clang_cc1 -triple x86_64-w64-windows-msvc  -emit-llvm -fobjc-runtime=gnustep-2.0 -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck --check-prefixes=CHECK-MSVC-OBJC2 < %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu  -emit-llvm -fobjc-runtime=gnustep-2.0 -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck --check-prefixes=CHECK-LINUX-OBJC2 < %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu  -emit-llvm -fobjc-runtime=gcc -fexceptions -fobjc-exceptions -o %t %s
// RUN: FileCheck --check-prefixes=CHECK-LINUX-GCC < %t %s
@interface Foo @end

void throwing(void) {
  @try
  {
    // CHECK-MINGW-OBJC2: personality ptr @__gxx_personality_seh0
    // CHECK-MINGW-OBJC2: invoke void @objc_exception_throw

    // CHECK-MINGW-GCC: personality ptr @__gnu_objc_personality_v0
    // CHECK-MINGW-GCC: invoke void @objc_exception_throw

    // CHECK-MSVC-OBJC2: personality ptr @__CxxFrameHandler3
    // CHECK-MSVC-OBJC2: invoke void @objc_exception_throw

    // CHECK-LINUX-OBJC2: personality ptr @__gnustep_objc_personality_v0
    // CHECK-LINUX-OBJC2: invoke void @objc_exception_throw

    // CHECK-LINUX-GCC: personality ptr @__gnu_objc_personality_v0
    @throw(@"error!");
  }
  @catch(...)
  {
    // CHECK-MINGW-OBJC2: call ptr @__cxa_begin_catch
    // CHECK-MINGW-OBJC2: invoke ptr @__cxa_rethrow
    // CHECK-MINGW-OBJC2: invoke void @__cxa_end_catch

    // CHECK-MINGW-GCC: call void @objc_exception_throw

    // CHECK-MSVC-OBJC2: call void @objc_exception_rethrow

    // CHECK-LINUX-OBJC2: call ptr @objc_begin_catch
    // CHECK-LINUX-OBJC2: invoke void @objc_exception_throw
    // CHECK-LINUX-OBJC2: invoke void @objc_end_catch()

    // CHECK-LINUX-GCC: invoke void @objc_exception_throw

    @throw;
  }
}
