// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG

// CIR: attributes {cir.global_ctors = [#cir.global_ctor<"__cxx_global_var_init", 101>]
// LLVM: @llvm.global_ctors = appending constant{{.*}}{ i32 101, ptr @__cxx_global_var_init, ptr null }
// OGCG: @llvm.global_ctors = appending global{{.*}}{ i32 101, ptr @_GLOBAL__I_000101, ptr null }
class A {
public:
  A(int, int);
} A __attribute((init_priority(101)))(0, 0);

// CIR-LABEL: cir.func internal private @__cxx_global_var_init() global_ctor(101)  {
// LLVM-LABEL: define internal void @__cxx_global_var_init() {
// OGCG-LABEL: define internal void @_GLOBAL__I_000101() {{.*}} section ".text.startup" {
