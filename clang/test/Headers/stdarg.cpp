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
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple le32-nacl -o - | FileCheck %s --check-prefix=PNACL-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple le32-nacl -o - | FileCheck %s --check-prefix=PNACL-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple i686-linux -o - | FileCheck %s --check-prefix=CHARPTR-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple i686-linux -o - | FileCheck %s --check-prefix=CHARPTR-CXX
// RUN: %clang_cc1 -emit-llvm -std=c99 -x c %s -triple xcore -o - | FileCheck %s --check-prefix=VOIDPTR-C
// RUN: %clang_cc1 -emit-llvm -std=c++17 -x c++ %s -triple xcore -o - | FileCheck %s --check-prefix=VOIDPTR-CXX

#include <stdarg.h>

// AARCH64-C: define {{.*}} @f(i32 %n, %struct.__va_list* %list)
// AARCH64-CXX: define {{.*}} @_Z1fiSt9__va_list(i32 %n, %"struct.std::__va_list"* %list)
// X86_64-C: define {{.*}} @f(i32 %n, %struct.__va_list_tag* %list)
// X86_64-CXX: define {{.*}} @_Z1fiP13__va_list_tag(i32 %n, %struct.__va_list_tag* %list)
// PPC64-C: define {{.*}} @f(i32 signext %n, i8* %list)
// PPC64-CXX: define {{.*}} @_Z1fiPc(i32 signext %n, i8* %list)
// AAPCS-C: define {{.*}} @f(i32 %n, [1 x i32] %list.coerce)
// AAPCS-CXX: define {{.*}} @_Z1fiSt9__va_list(i32 %n, [1 x i32] %list.coerce)
// SYSTEMZ-C: define {{.*}} @f(i32 signext %n, %struct.__va_list_tag* %list)
// SYSTEMZ-CXX: define {{.*}} @_Z1fiP13__va_list_tag(i32 signext %n, %struct.__va_list_tag* %list)
// PNACL-C: define {{.*}} @f(i32 %n, i32* %list)
// PNACL-CXX: define {{.*}} @_Z1fiPi(i32 %n, i32* %list)
// CHARPTR-C: define {{.*}} @f(i32 %n, i8* %list)
// CHARPTR-CXX: define {{.*}} @_Z1fiPc(i32 %n, i8* %list)
// VOIDPTR-C: define {{.*}} @f(i32 %n, i8* %list)
// VOIDPTR-CXX: define {{.*}} @_Z1fiPv(i32 %n, i8* %list)
void f(int n, va_list list) {}
