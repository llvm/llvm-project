// RUN: %clang_cc1 -triple powerpc64le-linux-unknown -target-cpu future %s \
// RUN:   -emit-llvm -o - | FileCheck %s

// CHECK: _Z2f1Pu13__vector_quad
void f1(__vector_quad *vq) {}

// CHECK: _Z2f2Pu13__vector_pair
void f2(__vector_pair *vp) {}

// CHECK: _Z2f3Pu13__vector_quad
void f3(__vector_quad *vq) {}

// CHECK: _Z2f3Pu13__vector_pair
void f3(__vector_pair *vp) {}
