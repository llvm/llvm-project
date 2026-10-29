// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,CIRLLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// Runtime library functions do not get "no-trapping-math" and
// "stack-protector-buffer-size", while user functions and compiler-generated
// init functions do, matching classic codegen.

struct S {
  S();
};

int f();

int g() {
  static int x = f();
  return x;
}

S s;

void h() { throw 1; }

// CIR: cir.func private @__cxa_guard_acquire({{.*}}) -> !s32i attributes {runtime_func}
// CIR: cir.func internal private @__cxx_global_var_init() {

// Classic codegen gives some runtime functions nounwind, which CIR does not
// do yet.
// CIRLLVM-DAG: declare i32 @__cxa_guard_acquire(ptr){{$}}
// CIRLLVM-DAG: declare ptr @__cxa_allocate_exception(i64){{$}}
// CIRLLVM-DAG: declare void @__cxa_throw(ptr, ptr, ptr){{$}}
// OGCG-DAG: declare i32 @__cxa_guard_acquire(ptr) #[[RT:[0-9]+]]
// OGCG-DAG: declare ptr @__cxa_allocate_exception(i64){{$}}
// OGCG-DAG: declare void @__cxa_throw(ptr, ptr, ptr){{$}}
// OGCG-DAG: attributes #[[RT]] = { nounwind }

// LLVM-DAG: define {{.*}}i32 @_Z1gv() #[[USER:[0-9]+]]
// LLVM-DAG: declare {{.*}}i32 @_Z1fv() #[[DECL:[0-9]+]]
// LLVM-DAG: define internal void @__cxx_global_var_init() #[[INIT:[0-9]+]]
// LLVM-DAG: define internal void @_GLOBAL__sub_I_{{.*}}() #[[INIT]]

// LLVM-DAG: attributes #[[USER]] = { {{.*}}"no-trapping-math"="true" "stack-protector-buffer-size"="8"
// LLVM-DAG: attributes #[[DECL]] = { {{.*}}"no-trapping-math"="true" "stack-protector-buffer-size"="8"
// LLVM-DAG: attributes #[[INIT]] = { {{.*}}"no-trapping-math"="true" "stack-protector-buffer-size"="8"
