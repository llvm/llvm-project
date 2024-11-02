// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o - %s | FileCheck %s
struct Coerce {
  id a;
};

struct Coerce coerce_func(void);

// CHECK-LABEL: define{{.*}} void @Coerce_test()
void Coerce_test(void) {
  struct Coerce c;
  
  // CHECK: call ptr @coerce_func
  // CHECK: call ptr @objc_memmove_collectable(
  c = coerce_func();
}

struct Indirect {
  id a;
  int b[10];
};

struct Indirect indirect_func(void);

// CHECK-LABEL: define{{.*}} void @Indirect_test()
void Indirect_test(void) {
  struct Indirect i;
  
  // CHECK: call void @indirect_func(ptr dead_on_unwind writable sret
  // CHECK: call ptr @objc_memmove_collectable(
  i = indirect_func();
}
