// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple aarch64-linux -o - | FileCheck %s --check-prefix=AARCH64-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple aarch64-linux -o - | FileCheck %s --check-prefix=AARCH64-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple x86_64-linux -o - | FileCheck %s --check-prefix=X86_64-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple x86_64-linux -o - | FileCheck %s --check-prefix=X86_64-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple ppc64-linux -o - | FileCheck %s --check-prefix=PPC64-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple ppc64-linux -o - | FileCheck %s --check-prefix=PPC64-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple armv7-apple-darwin9 -target-abi aapcs -o - | FileCheck %s --check-prefix=AAPCS-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple armv7-apple-darwin9 -target-abi aapcs -o - | FileCheck %s --check-prefix=AAPCS-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple s390x-linux -o - | FileCheck %s --check-prefix=SYSTEMZ-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple s390x-linux -o - | FileCheck %s --check-prefix=SYSTEMZ-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple i686-linux -o - | FileCheck %s --check-prefix=CHARPTR-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple i686-linux -o - | FileCheck %s --check-prefix=CHARPTR-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple xcore -o - | FileCheck %s --check-prefix=VOIDPTR-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple xcore -o - | FileCheck %s --check-prefix=VOIDPTR-CXX

#include <stdarg.h>

// AARCH64-C: define {{.*}} @f(i32 noundef %n, %struct.__va_list* noundef %list)
// AARCH64-CXX: define {{.*}} @_Z1fiSt9__va_list(i32 noundef %n, %"struct.std::__va_list"* noundef %list)
// X86_64-C: define {{.*}} @f(i32 noundef %n, %struct.__va_list_tag* noundef %list)
// X86_64-CXX: define {{.*}} @_Z1fiP13__va_list_tag(i32 noundef %n, %struct.__va_list_tag* noundef %list)
// PPC64-C: define {{.*}} @f(i32 noundef signext %n, i8* noundef %list)
// PPC64-CXX: define {{.*}} @_Z1fiPc(i32 noundef signext %n, i8* noundef %list)
// AAPCS-C: define {{.*}} @f(i32 noundef %n, [1 x i32] %list.coerce)
// AAPCS-CXX: define {{.*}} @_Z1fiSt9__va_list(i32 noundef %n, [1 x i32] %list.coerce)
// SYSTEMZ-C: define {{.*}} @f(i32 noundef signext %n, %struct.__va_list_tag* noundef %list)
// SYSTEMZ-CXX: define {{.*}} @_Z1fiP13__va_list_tag(i32 noundef signext %n, %struct.__va_list_tag* noundef %list)
// CHARPTR-C: define {{.*}} @f(i32 noundef %n, i8* noundef %list)
// CHARPTR-CXX: define {{.*}} @_Z1fiPc(i32 noundef %n, i8* noundef %list)
// VOIDPTR-C: define {{.*}} @f(i32 noundef %n, i8* noundef %list)
// VOIDPTR-CXX: define {{.*}} @_Z1fiPv(i32 noundef %n, i8* noundef %list)
void f(int n, va_list list) {}
