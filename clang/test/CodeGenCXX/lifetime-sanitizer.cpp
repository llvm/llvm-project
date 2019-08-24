// RUN: %clang -w -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 %s | \
// RUN:      FileCheck %s -check-prefixes=CHECK,CHECK-O0 --implicit-check-not=llvm.lifetime
// RUN: %clang -w -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// RUN:     -fsanitize=address -fsanitize-address-use-after-scope %s | \
// RUN:     FileCheck %s -check-prefixes=CHECK,LIFETIME
// REQUIRES: assertions

extern int bar(char *A, int n);

struct X {
  X();
  ~X();
  int *p;
};
struct Y {
  Y();
  int *p;
};

extern "C" void a(), b(), c(), d();

// CHECK-LABEL: @_Z3foo
void foo(int n) {
  // CHECK-LABEL: call void @a()
  a();

  // CHECK-LABEL: call void @b()
  // CHECK: store i1 false
  // CHECK-LABEL: br i1
  //
  // CHECK-LABEL: cond.true:
  // LIFETIME: @llvm.lifetime.start
  // LIFETIME: store i1 true
  // LIFETIME: call void @_ZN1XC
  // CHECK-LABEL: br label
  //
  // CHECK-LABEL: cond.false:
  // LIFETIME: @llvm.lifetime.start
  // LIFETIME: store i1 true
  // LIFETIME: call void @_ZN1YC
  // CHECK-LABEL: br label
  //
  // CHECK-LABEL: cond.end:
  // CHECK: call void @c()
  // LIFETIME: @llvm.lifetime.end
  // LIFETIME: @llvm.lifetime.end
  b(), (n ? X().p : Y().p), c();

  // CHECK: call void @d()
  d();
}
