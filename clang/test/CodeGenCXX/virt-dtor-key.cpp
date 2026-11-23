// RUN: %clang_cc1 -triple i686-linux         -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i686-windows-gnu   -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MINGW
// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MINGW
// RUN: %clang_cc1 -triple i686-pc-cygwin     -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MINGW
// RUN: %clang_cc1 -triple x86_64-pc-cygwin   -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-MINGW

// CHECK: @_ZTI3foo ={{.*}} constant
// CHECK-MINGW: @_ZTI3foo = linkonce_odr
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
