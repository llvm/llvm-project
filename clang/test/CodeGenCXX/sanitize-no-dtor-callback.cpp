// Test with the flag -fno-sanitize-memory-use-after-dtor, to ensure that
// instrumentation is not erroneously inserted
// RUN: %clang_cc1 -fsanitize=memory -fno-sanitize-memory-use-after-dtor -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

struct Simple {
  int x;
  ~Simple() {}
};
Simple s;
// CHECK-LABEL: define {{.*}}SimpleD1Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback

struct Inlined {
  int x;
  inline ~Inlined() {}
};
Inlined i;
// CHECK-LABEL: define {{.*}}InlinedD1Ev

// CHECK-LABEL: define {{.*}}SimpleD2Ev

// CHECK-LABEL: define {{.*}}InlinedD2Ev
