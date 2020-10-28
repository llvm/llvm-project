// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -emit-llvm %s -o - | FileCheck %s

typedef signed long long V2LLi __attribute__((vector_size(16)));
typedef signed long long V4LLi __attribute__((vector_size(32)));

V2LLi ret_128();
V4LLi ret_256();
void arg_128(V2LLi);
void arg_256(V4LLi);

// Make sure return type forces a min-legal-width
V2LLi foo(void) {
  return (V2LLi){ 0, 0 };
}

V4LLi goo(void) {
  return (V4LLi){ 0, 0 };
}

// Make sure return type of called function forces a min-legal-width
void hoo(void) {
  V2LLi tmp_V2LLi;
  tmp_V2LLi = ret_128();
}

void joo(void) {
  V4LLi tmp_V4LLi;
  tmp_V4LLi = ret_256();
}

// Make sure arg type of called function forces a min-legal-width
void koo(void) {
  V2LLi tmp_V2LLi;
  arg_128(tmp_V2LLi);
}

void loo(void) {
  V4LLi tmp_V4LLi;
  arg_256(tmp_V4LLi);
}

// Make sure arg type of our function forces a min-legal-width
void moo(V2LLi x) {

}

void noo(V4LLi x) {

}

// CHECK: {{.*}}@foo{{.*}} #0
// CHECK: {{.*}}@goo{{.*}} #1
// CHECK: {{.*}}@hoo{{.*}} #0
// CHECK: {{.*}}@joo{{.*}} #1
// CHECK: {{.*}}@koo{{.*}} #0
// CHECK: {{.*}}@loo{{.*}} #1
// CHECK: {{.*}}@moo{{.*}} #0
// CHECK: {{.*}}@noo{{.*}} #1

// CHECK: #0 = {{.*}}"min-legal-vector-width"="128"
// CHECK: #1 = {{.*}}"min-legal-vector-width"="256"
