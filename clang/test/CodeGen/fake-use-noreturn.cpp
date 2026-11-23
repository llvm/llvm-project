// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -o - | FileCheck %s
//
// Check we can correctly produce fake uses for function-level variables even
// when we have a return in a nested conditional and there is no code at the end
// of the function.

// CHECK-LABEL: define{{.*}}@_Z3fooi
// CHECK:         [[I_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr %i.addr
// CHECK:         call void (...) @llvm.fake.use(i32 [[I_FAKE_USE]])
// CHECK-LABEL: define{{.*}}@_ZN1C3barEi
// CHECK:         [[J_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr %j.addr
// CHECK:         call void (...) @llvm.fake.use(i32 [[J_FAKE_USE]])

void foo(int i) {
   while (0)
     if (1)
       return;
}

class C {
  void bar(int j);
};

void C::bar(int j) {
  while (0)
    if (1)
      return;
}
