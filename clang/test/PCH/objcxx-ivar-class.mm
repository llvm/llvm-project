// UNSUPPORTED: -zos, -aix
// Test this without pch.
// RUN: %clang_cc1 -no-opaque-pointers -include %S/objcxx-ivar-class.h -triple %itanium_abi_triple %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -no-opaque-pointers -x objective-c++-header -triple %itanium_abi_triple -emit-pch -o %t %S/objcxx-ivar-class.h
// RUN: %clang_cc1 -no-opaque-pointers -include-pch %t -triple %itanium_abi_triple %s -emit-llvm -o - | FileCheck %s

// CHECK: [C position]
// CHECK: call {{.*}} @_ZN1SC1ERKS_

// CHECK: [C setPosition:]
// CHECK: = call {{.*}}%struct.S* @_ZN1SaSERKS_

// CHECK: [C .cxx_destruct]
// CHECK: [C .cxx_construct]
