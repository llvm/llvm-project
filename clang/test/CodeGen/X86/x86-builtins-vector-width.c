// RUN: %clang_cc1 -triple i686-linux-gnu -target-cpu i686 -emit-llvm %s -o - | FileCheck %s

typedef double V2LLi __attribute__((vector_size(16)));
typedef double V4LLi __attribute__((vector_size(32)));

// Make sure builtin forces a min-legal-width attribute
void foo(void) {
  V2LLi  tmp_V2LLi;

  tmp_V2LLi = __builtin_ia32_undef128();
}

// Make sure explicit attribute larger than builtin wins.
void goo(void) __attribute__((__min_vector_width__(256))) {
  V2LLi  tmp_V2LLi;

  tmp_V2LLi = __builtin_ia32_undef128();
}

// Make sure builtin larger than explicit attribute wins.
void hoo(void) __attribute__((__min_vector_width__(128))) {
  V4LLi  tmp_V4LLi;

  tmp_V4LLi = __builtin_ia32_undef256();
}

// CHECK: foo{{.*}} #0
// CHECK: goo{{.*}} #1
// CHECK: hoo{{.*}} #1

// CHECK: #0 = {{.*}}"min-legal-vector-width"="128"
// CHECK: #1 = {{.*}}"min-legal-vector-width"="256"
