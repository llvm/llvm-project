// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=kcfi -fsanitize-trap=kcfi -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=UNGENERALIZED %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=kcfi -fsanitize-trap=kcfi -fsanitize-cfi-icall-generalize-pointers -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=GENERALIZED %s

// Test that const char* is generalized to const ptr and that char** is
// generalized to ptr

// CHECK: define{{.*}} ptr @f({{.*}} !kcfi_type [[TYPE:![0-9]+]]
int** f(const char *a, const char **b) {
  return (int**)0;
}

// GENERALIZED: define{{.*}} ptr @f2({{.*}} !kcfi_type [[TYPE]]
// UNGENERALIZED: define{{.*}} ptr @f2({{.*}} !kcfi_type [[TYPE2:![0-9]+]]
int** f2(const int *a, const int **b) {
  return (int**)0;
}

// CHECK: define{{.*}} ptr @f3({{.*}} !kcfi_type [[TYPE3:![0-9]+]]
int** f3(char *a, char **b) {
  return (int**)0;
}

void g(int** (*fp)(const char *, const char **)) {
  // UNGENERALIZED: call {{.*}} [ "kcfi"(i32 1296635908) ]
  // GENERALIZED: call {{.*}} [ "kcfi"(i32 -49168686) ]
  fp(0, 0);
}

union Union {
  char *c;
  long *n;
} __attribute__((transparent_union));

// CHECK: define{{.*}} void @uni({{.*}} !kcfi_type [[TYPE4:![0-9]+]]
void uni(void (*fn)(union Union), union Union arg1) {
  // UNGENERALIZED: call {{.*}} [ "kcfi"(i32 -1037059548) ]
  // GENERALIZED: call {{.*}} [ "kcfi"(i32 422130955) ]
    fn(arg1);
}

// UNGENERALIZED: [[TYPE]] = !{i32 1296635908}
// GENERALIZED: [[TYPE]] = !{i32 -49168686}

// UNGENERALIZED: [[TYPE3]] = !{i32 874141567}
// GENERALIZED: [[TYPE3]] = !{i32 954385378}

// UNGENERALIZED: [[TYPE4]] = !{i32 981319178}
// GENERALIZED: [[TYPE4]] = !{i32 -1599950473}

