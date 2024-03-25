// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O0 -emit-llvm -o - %s | FileCheck %s

void use(void *);

void test_small(void) {
  // CHECK-LABEL: define{{.*}} void @test_small()
  int a[] = {1, 2, 3, 4};
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}

void test_small_same(void) {
  // CHECK-LABEL: define{{.*}} void @test_small_same()
  char a[] = {'a', 'a', 'a', 'a'};
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}

void test_different(void) {
  // CHECK-LABEL: define{{.*}} void @test_different()
  int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}

void test_all_zeros(void) {
  // CHECK-LABEL: define{{.*}} void @test_all_zeros()
  int a[16] = {};
  // CHECK: call void @llvm.memset.{{.*}}
  use(a);
}

void test_all_a(void) {
  // CHECK-LABEL: define{{.*}} void @test_all_a()
  char a[] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}

void test_most_zeros(void) {
  // CHECK-LABEL: define{{.*}} void @test_most_zeros()
  int a[16] = {0, 0, 1};
  // CHECK: call void @llvm.memset.{{.*}}
  // CHECK: store i32 1
  use(a);
}

void test_most_a(void) {
  // CHECK-LABEL: define{{.*}} void @test_most_a()
  char a[] = "aaaaazaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}

void test_pointers(void) {
  // CHECK-LABEL: define{{.*}} void @test_pointers()
  void *a[] = {&use, &use, &use, &use, &use, &use};
  // CHECK: call void @llvm.memcpy.{{.*}}
  use(a);
}
