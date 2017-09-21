// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

// Sanitizing dtor is emitted in dtor for every class, and only
// poisons once.

struct Simple {
  int x;
  ~Simple() {}
};
Simple s;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK-LABEL: define {{.*}}SimpleD1Ev
// CHECK: call void {{.*}}SimpleD2Ev
// CHECK: ret void

struct Inlined {
  int y;
  inline ~Inlined() {}
};
Inlined i;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK-LABEL: define {{.*}}InlinedD1Ev
// CHECK: call void {{.*}}InlinedD2Ev
// CHECK: ret void

struct Defaulted_Trivial {
  ~Defaulted_Trivial() = default;
};
void create_def_trivial() {
  Defaulted_Trivial def_trivial;
}
// The compiler is explicitly signalled to handle object cleanup.
// No complex member attributes. Compiler destroys inline, so
// no destructor defined.
// CHECK-LABEL: define {{.*}}create_def_trivial
// CHECK-NOT: call {{.*}}Defaulted_Trivial
// CHECK: ret void

struct Defaulted_Non_Trivial {
  Simple s;
  ~Defaulted_Non_Trivial() = default;
};
Defaulted_Non_Trivial def_non_trivial;
// Explicitly compiler-generated dtor poisons object.
// By including a Simple member in the struct, the compiler is
// forced to generate a non-trivial destructor.
// CHECK-LABEL: define {{.*}}Defaulted_Non_TrivialD1Ev
// CHECK: call void {{.*}}Defaulted_Non_TrivialD2
// CHECK: ret void


// Note: ordering is important. In the emitted bytecode, these
// second dtors defined after the first. Explicitly checked here
// to confirm that all invoked dtors have member poisoning
// instrumentation inserted.
// CHECK-LABEL: define {{.*}}SimpleD2Ev
// CHECK-NOT: store i{{[0-9]+}} 0, {{.*}}@__msan_param_tls
// CHECK: call void @__sanitizer_dtor_callback
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void

// CHECK-LABEL: define {{.*}}InlinedD2Ev
// CHECK-NOT: store i{{[0-9]+}} 0, {{.*}}@__msan_param_tls
// CHECK: call void @__sanitizer_dtor_callback
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void

// CHECK-LABEL: define {{.*}}Defaulted_Non_TrivialD2Ev
// CHECK-NOT: store i{{[0-9]+}} 0, {{.*}}@__msan_param_tls
// CHECK: call void @__sanitizer_dtor_callback
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void
