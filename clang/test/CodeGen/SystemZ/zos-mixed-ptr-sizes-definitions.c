// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm < %s | FileCheck %s --check-prefix=PTR32-ZOS
// RUN: %clang_cc1 -triple s390x-ibm-linux -fzos-extensions -emit-llvm < %s | FileCheck %s --check-prefix=PTR32-LINUX
// RUN: %clang_cc1 -triple s390x-linux-gnu -fzos-extensions -emit-llvm < %s | FileCheck %s --check-prefix=PTR32-LINUX

void ptr32_declarations() {
  // PTR32-ZOS-LABEL: @ptr32_declarations()
  // PTR32-LINUX-LABEL: @ptr32_declarations()

  // PTR32-ZOS: %p1 = alloca ptr addrspace(1), align 4
  // PTR32-LINUX-NOT: %p1 = alloca i8 addrspace(1)*, align 4
  // PTR32-LINUX: %p1 = alloca ptr, align 8
  char * __ptr32 p1;

  // PTR32-ZOS: %p2 = alloca ptr, align 8
  // PTR32-LINUX-NOT: %p2 = alloca ptr addrspace(1), align 8
  // PTR32-LINUX: %p2 = alloca ptr, align 8
  char * __ptr32 *p2;

  // PTR32-ZOS: %p3 = alloca ptr addrspace(1), align 4
  // PTR32-LINUX-NOT: %p3 = alloca i8* addrspace(1)*, align 4
  // PTR32-LINUX: %p3 = alloca ptr, align 8
  char ** __ptr32 p3;

  // PTR32-ZOS: %p4 = alloca ptr, align 8
  // PTR32-LINUX-NOT: %p4 = alloca ptr addrspace(1), align 8
  // PTR32-LINUX: %p4 = alloca ptr, align 8
  char ** __ptr32 *p4;

  // PTR32-ZOS: %p5 = alloca ptr, align 8
  // PTR32-LINUX-NOT:  %p5 = alloca ptr addrspace(1), align 8
  // PTR32-LINUX: %p5 = alloca ptr, align 8
  char *** __ptr32 *p5;

  // PTR32-ZOS: %p6 = alloca ptr, align 8
  // PTR32-LINUX: %p6 = alloca ptr, align 8
  char **p6;

  // PTR32-ZOS: %p7 = alloca ptr addrspace(1), align 4
  // PTR32-LINUX-NOT: %p7 = alloca i8 addrspace(1)* addrspace(1)*, align 4
  // PTR32-LINUX: %p7 = alloca ptr, align 8
  char * __ptr32 * __ptr32 p7;

  // PTR32-ZOS: %p8 = alloca ptr addrspace(1), align 4
  // PTR32-LINUX-NOT: %p8 = alloca i8* addrspace(1)* addrspace(1)*, align 4
  // PTR32-LINUX: %p8 = alloca ptr, align 8
  char ** __ptr32 * __ptr32 p8;

  // PTR32-ZOS: %p9 = alloca ptr, align 8
  // PTR32-LINUX-NOT: %p9 = alloca i8* addrspace(1)* addrspace(1)**, align 8
  // PTR32-LINUX: %p9 = alloca ptr, align 8
  char ** __ptr32 * __ptr32 *p9;

}
