// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct A {
  float elements[4];
  A(float const *src);
};

// CHECK: define {{.*}} @_ZN1AC1EPKf(ptr {{.*}}noalias{{.*}} %this, {{.*}})
A::A(float const *src) {
  elements[0] = src[0];
  elements[1] = src[1];
  elements[2] = src[2];
  elements[3] = src[3];
}
