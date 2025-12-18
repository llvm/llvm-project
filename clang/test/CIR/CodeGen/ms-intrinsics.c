// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - \
// RUN:   | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - \
// RUN:   | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/ms-intrinsics.c, which eventually
// CIR shall be able to support fully.

void *_AddressOfReturnAddress(void);
void *_ReturnAddress(void);

void *test_ReturnAddress(void) {
  return _ReturnAddress();
  // CIR-LABEL: test_ReturnAddress
  // CIR: [[ARG:%.*]] = cir.const #cir.int<0> : !u32i
  // CIR: {{%.*}} = cir.return_address([[ARG]])

  // LLVM-LABEL: test_ReturnAddress
  // LLVM: {{%.*}} = call ptr @llvm.returnaddress(i32 0)

  // OGCG-LABEL: test_ReturnAddress
  // OGCG: {{%.*}} = call ptr @llvm.returnaddress(i32 0)
}

#if defined(__i386__) || defined(__x86_64__) || defined (__aarch64__)
void *test_AddressOfReturnAddress(void) {
  return _AddressOfReturnAddress();
  // CIR-LABEL: test_AddressOfReturnAddress
  // CIR: %[[ADDR:.*]] = cir.address_of_return_address : !cir.ptr<!u8i>
  // CIR: %{{.*}} = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>

  // LLVM-LABEL: test_AddressOfReturnAddress
  // LLVM: call ptr @llvm.addressofreturnaddress.p0()

  // OGCG-LABEL: test_AddressOfReturnAddress
  // OGCG: call ptr @llvm.addressofreturnaddress.p0()
}
#endif
