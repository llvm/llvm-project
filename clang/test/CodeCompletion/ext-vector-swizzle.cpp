// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:9:7 %s -o - | FileCheck --check-prefix=CHECK-VEC4 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:17:7 %s -o - | FileCheck --check-prefix=CHECK-VEC2 %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));

void test() {
  float4 pos;
  pos.
}
// CHECK-VEC4-DAG: COMPLETION: Pattern : x
// CHECK-VEC4-DAG: COMPLETION: Pattern : xyzw
// CHECK-VEC4-DAG: COMPLETION: Pattern : rgba

void test2() {
  float2 tex;
  tex.
}
// CHECK-VEC2-DAG: COMPLETION: Pattern : x
// CHECK-VEC2-DAG: COMPLETION: Pattern : y
// CHECK-VEC2-NOT: COMPLETION: Pattern : z
// CHECK-VEC2-NOT: COMPLETION: Pattern : w
