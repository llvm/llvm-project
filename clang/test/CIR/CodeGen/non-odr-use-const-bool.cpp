// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Foo { static const bool flag = true; };

void take_bool(bool);
extern int side();

void pass_to_call(Foo x) {
  take_bool(x.flag);
}

// CIR-LABEL: cir.func{{.*}} @_Z12pass_to_call3Foo
// CIR:         %[[B_CALL:.+]] = cir.const #true
// CIR:         cir.call @_Z9take_boolb(%[[B_CALL]]) : (!cir.bool{{.*}}) -> ()

// LLVM-LABEL: define {{.*}} void @_Z12pass_to_call3Foo
// LLVM:         call void @_Z9take_boolb(i1 {{.*}}true)

// OGCG-LABEL: define {{.*}} void @_Z12pass_to_call3Foo
// OGCG:         call void @_Z9take_boolb(i1 {{.*}}true)

int use_in_if(Foo x) {
  if (x.flag) return 1;
  return 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z9use_in_if3Foo
// CIR:         %[[B_IF:.+]] = cir.const #true
// CIR:         cir.if %[[B_IF]]

// LLVM-LABEL: define {{.*}}i32 @_Z9use_in_if3Foo
// LLVM:         br i1 true,

// OGCG-LABEL: define {{.*}}i32 @_Z9use_in_if3Foo

int short_circuit(Foo x) {
  return (x.flag && side()) ? 1 : 0;
}

// CIR-LABEL: cir.func{{.*}} @_Z13short_circuit3Foo
// CIR:         %[[B_TERN:.+]] = cir.const #true
// CIR:         cir.ternary(%[[B_TERN]],

// LLVM-LABEL: define {{.*}}i32 @_Z13short_circuit3Foo

// OGCG-LABEL: define {{.*}}i32 @_Z13short_circuit3Foo
