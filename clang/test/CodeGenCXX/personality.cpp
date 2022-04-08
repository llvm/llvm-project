// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=dwarf -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-DWARF
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=seh -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-linux-gnu -fexceptions -exception-model=sjlj -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-SJLJ

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X86
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-windows-msvc -D __SEH_EXCEPTIONS__ -fms-extensions -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-WIN-SEH-X64

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=dwarf -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-DWARF
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=seh -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-SEH
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-unknown-windows-gnu -fexceptions -exception-model=sjlj -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-GNU-SJLJ

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix-xcoff -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-AIX
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-aix-xcoff -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-AIX

extern void g();

// CHECK-GNU: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// CHECK-GNU-DWARF: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// CHECK-GNU-SEH: personality i8* bitcast (i32 (...)* @__gxx_personality_seh0 to i8*)
// CHECK-GNU-SJLJ: personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)

// CHECK-WIN: personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)

// CHECK-AIX: personality i8* bitcast (i32 (...)* @__xlcxx_personality_v1 to i8*)

void f() {
  try {
    g();
  } catch (...) {
  }
}

#if defined(__SEH_EXCEPTIONS__)
// CHECK-WIN-SEH-X86: personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK-WIN-SEH-X64: personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)

void h(void) {
  __try {
    g();
  } __finally {
  }
}
#endif

