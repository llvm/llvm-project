// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature +sse2 < %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-feature -sse2 < %s | FileCheck %s --check-prefixes=CHECK

struct st1 {
  __float128 a;
};

struct st1 h1(__float128 a) {
  // CHECK: define{{.*}}fp128 @h1(fp128
  struct st1 x;
  x.a = a;
  return x;
}

__float128 h2(struct st1 x) {
  // CHECK: define{{.*}}fp128 @h2(fp128
  return x.a;
}

struct st2 {
  __float128 a;
  int b;
};

struct st2 h3(__float128 a, int b) {
  // CHECK: define{{.*}}void @h3(ptr {{.*}}sret(%struct.st2)
  struct st2 x;
  x.a = a;
  x.b = b;
  return x;
}

__float128 h4(struct st2 x) {
  // CHECK: define{{.*}}fp128 @h4(ptr {{.*}}byval(%struct.st2)
  return x.a;
}
