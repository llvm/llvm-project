// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_AddressOfReturnAddress

// _AddressOfReturnAddress is the address of the slot that holds the return
// address.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

void *test_AddressOfReturnAddress(void) { return _AddressOfReturnAddress(); }
// CHECK-LABEL: define dso_local {{.*}}ptr @test_AddressOfReturnAddress()
// CHECK: [[R:%.*]] = call ptr @llvm.addressofreturnaddress.p0()
// CHECK: ret ptr [[R]]
