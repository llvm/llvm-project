// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=both %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=clang %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-CLANG
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=platform %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-PLATFORM
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=both -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=clang -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-CLANG
// RUN: %clang_cc1 -triple x86_64-linux-gnu -msmall-int-abi-compat=platform -fexperimental-abi-lowering %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-PLATFORM

// -msmall-int-abi-compat=platform only needs to extend to 8-bits in the caller
// and can assume 8-bit extension in the callee. We currently can't express
// this, so we perform full 32-bit extension in the caller and do not assume
// any extension in the callee, making it the same as =both for the special
// case of _Bool. In the future, we should model this more precisely.

// CHECK-BOTH: define{{.*}} zeroext i1 @test(i1 noundef %x)
// CHECK-CLANG: define{{.*}} zeroext i1 @test(i1 noundef zeroext %x)
// CHECK-PLATFORM: define{{.*}} zeroext i1 @test(i1 noundef %x)
_Bool test(_Bool x) {
  return x;
}

_Bool test2(_Bool x) {
// CHECK-BOTH: call zeroext i1 @test(i1 noundef zeroext %{{.*}})
// CHECK-CLANG: call zeroext i1 @test(i1 noundef zeroext %{{.*}})
// CHECK-PLATFORM: call zeroext i1 @test(i1 noundef zeroext %{{.*}})
  return test(x);
}
