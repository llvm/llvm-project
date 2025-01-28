// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -o - | FileCheck %s --implicit-check-not=llvm.fake.use
// Check that fake use calls are emitted at the correct locations, i.e.
// at the end of lexical blocks and at the end of the function.

int glob_i;
char glob_c;
float glob_f;

int foo(int i) {
  // CHECK-LABEL: define{{.*}}foo
  if (i < 4) {
    char j = i * 3;
    if (glob_i > 3) {
      float f = glob_f;
      j = f;
      glob_c = j;
      // CHECK: call void (...) @llvm.fake.use(float %
      // CHECK-NEXT: br label %
    }
    glob_i = j;
    // CHECK: call void (...) @llvm.fake.use(i8 %
    // CHECK-NEXT: br label %
  }
  // CHECK: call void (...) @llvm.fake.use(i32 %
  // CHECK-NEXT: ret
  return 4;
}

// CHECK: declare void @llvm.fake.use(...)
