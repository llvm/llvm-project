// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 %s | FileCheck %s -check-prefixes=CHECK,CHECK-O0 --implicit-check-not=llvm.lifetime
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// RUN:     -fsanitize=address -fsanitize-address-use-after-scope %s | \
// RUN:     FileCheck %s -check-prefixes=CHECK,CHECK-ASAN-USE-AFTER-SCOPE

extern int bar(char *A, int n);

struct X { X(); ~X(); int *p; };
struct Y { Y(); int *p; };

extern "C" void a(), b(), c(), d();

// CHECK-LABEL: @_Z3foo
void foo(int n) {
  // CHECK: call void @a()
  a();

  // CHECK: call void @b()
  // CHECK-ASAN-USE-AFTER-SCOPE: store i1 false
  // CHECK-ASAN-USE-AFTER-SCOPE: store i1 false
  // CHECK: br i1
  //
  // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.start
  // CHECK-ASAN-USE-AFTER-SCOPE: store i1 true
  // CHECK: call void @_ZN1XC
  // CHECK: br label
  //
  // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.start
  // CHECK-ASAN-USE-AFTER-SCOPE: store i1 true
  // CHECK: call void @_ZN1YC
  // CHECK: br label
  //
  // CHECK: call void @c()
  // CHECK-ASAN-USE-AFTER-SCOPE: br i1
  // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.end
  // CHECK-ASAN-USE-AFTER-SCOPE: br i1
  // CHECK-ASAN-USE-AFTER-SCOPE: @llvm.lifetime.end
  b(), (n ? X().p : Y().p), c();

  // CHECK: call void @d()
  d();
}
