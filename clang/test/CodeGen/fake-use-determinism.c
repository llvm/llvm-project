// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness %s -o - | FileCheck %s
//
// We are checking that the fake.use calls for i, j and k appear
// in a particular order. It is not the order itself that is important
// but that it remains the same between different test runs.

// CHECK:      [[K_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr %k.addr
// CHECK-NEXT: call void (...) @llvm.fake.use(i32 [[K_FAKE_USE]]) #2
// CHECK-NEXT: [[J_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr %j.addr
// CHECK-NEXT: call void (...) @llvm.fake.use(i32 [[J_FAKE_USE]]) #2
// CHECK-NEXT: [[I_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr %i.addr
// CHECK-NEXT: call void (...) @llvm.fake.use(i32 [[I_FAKE_USE]]) #2

void bar();
void foo(int i, int j, int k)
{
   for (int l = 0; l < i; l++) {
      bar();
   }
}
