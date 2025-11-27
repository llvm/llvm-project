// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s


void test_builtins_basic() {
  __builtin_operator_delete(__builtin_operator_new(4));
  // CIR-LABEL: test_builtins_basic
  // CIR: [[P:%.*]] = cir.call @_Znwm({{%.*}}) : (!u64i) -> !cir.ptr<!void>
  // CIR: cir.call @_ZdlPv([[P]]) {{.*}}: (!cir.ptr<!void>) -> ()
  // CIR: cir.return

  // LLVM-LABEL: test_builtins_basic
  // LLVM: [[P:%.*]] = call ptr @_Znwm(i64 4)
  // LLVM: call void @_ZdlPv(ptr [[P]])
  // LLVM: ret void

  // OGCG-LABEL: test_builtins_basic
  // OGCG: [[P:%.*]] = call {{.*}} ptr @_Znwm(i64 {{.*}} 4)
  // OGCG: call void @_ZdlPv(ptr {{.*}} [[P]])
  // OGCG: ret void
}

void test_sized_delete() {
  __builtin_operator_delete(__builtin_operator_new(4), 4);

  // CIR-LABEL: test_sized_delete
  // CIR: [[P:%.*]] = cir.call @_Znwm({{%.*}}) : (!u64i) -> !cir.ptr<!void>
  // CIR: cir.call @_ZdlPvm([[P]], {{%.*}}) {{.*}}: (!cir.ptr<!void>, !u64i) -> ()
  // CIR: cir.return

  // LLVM-LABEL: test_sized_delete
  // LLVM: [[P:%.*]] = call ptr @_Znwm(i64 4)
  // LLVM: call void @_ZdlPvm(ptr [[P]], i64 4)
  // LLVM: ret void

  // OGCG-LABEL: test_sized_delete
  // OGCG: [[P:%.*]] = call {{.*}} ptr @_Znwm(i64 {{.*}} 4)
  // OGCG: call void @_ZdlPvm(ptr {{.*}} [[P]], i64 {{.*}} 4)
  // OGCG: ret void
}
