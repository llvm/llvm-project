// Global initialization and cleanup functions are called through function
// pointers from .init_array, .fini_array and atexit registrations, so they get
// call graph type IDs like any other address-taken function, matching the type
// of the pointer they are registered through.

// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -emit-llvm -o - \
// RUN:   -fexperimental-call-graph-section %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -emit-llvm -o - \
// RUN:   -fexperimental-call-graph-section \
// RUN:   -fsanitize-cfi-icall-experimental-normalize-integers %s | \
// RUN:   FileCheck %s --check-prefix=NORM
// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -emit-llvm -o - \
// RUN:   -fexperimental-call-graph-section -fno-use-cxa-atexit %s | \
// RUN:   FileCheck %s --check-prefix=ATEXIT

struct S {
  S();
  ~S();
};

S Global;
S Array[2];

// CHECK-DAG: define internal void @__cxx_global_var_init() {{.*}}!callgraph [[VOID:![0-9]+]]
// CHECK-DAG: define internal void @__cxx_global_array_dtor(ptr {{.*}}) {{.*}}!callgraph [[VOIDPTR:![0-9]+]]
// CHECK-DAG: define internal void @_GLOBAL__sub_I_{{.*}}() {{.*}}!callgraph [[VOID]]
// CHECK-DAG: [[VOID]] = !{!"_ZTSFvvE"}
// CHECK-DAG: [[VOIDPTR]] = !{!"_ZTSFvPvE"}

// NORM-DAG: define internal void @_GLOBAL__sub_I_{{.*}}() {{.*}}!callgraph [[VOIDN:![0-9]+]]
// NORM-DAG: [[VOIDN]] = !{!"_ZTSFvvE.normalized"}

// ATEXIT-DAG: define internal void @__dtor_Global() {{.*}}!callgraph [[VOID:![0-9]+]]
// ATEXIT-DAG: [[VOID]] = !{!"_ZTSFvvE"}
