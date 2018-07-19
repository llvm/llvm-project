// Test without the flag -fsanitize-memory-use-after-dtor, to ensure that
// instrumentation is not erroneously inserted
// RUN: %clang_cc1 -fsanitize=memory -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

struct Simple {
  ~Simple() {}
};
Simple s;
// CHECK-LABEL: define {{.*}}SimpleD1Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback

struct Inlined {
  inline ~Inlined() {}
};
Inlined i;
// CHECK-LABEL: define {{.*}}InlinedD1Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback

// CHECK-LABEL: define {{.*}}SimpleD2Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback

// CHECK-LABEL: define {{.*}}InlinedD2Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback
