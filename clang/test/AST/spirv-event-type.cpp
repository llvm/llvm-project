// REQUIRES: spirv-registered-target
// Test without serialization:
// RUN: %clang_cc1 -triple spirv64 -ast-dump %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple spirv64 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple spirv64 -include-pch %t -ast-dump-all /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s

void test(void) {
  __spirv_event_t e;
}

// CHECK: TypedefDecl {{.*}} implicit {{.*}}__spirv_event_t '__spirv_event_t'
// CHECK-NEXT: -BuiltinType {{.*}} '__spirv_event_t'
// CHECK: VarDecl {{.*}} e '__spirv_event_t'
