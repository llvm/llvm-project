// expected-no-diagnostics
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int foo_variant_dispatch(int x, int y) {
   return x+2;
}

int foo_variant_allCond(int x, int y) {
   return x+3;
}

#pragma omp declare variant(foo_variant_dispatch) match(construct={dispatch})
#pragma omp declare variant(foo_variant_allCond) match(user={condition(1)})
int foo(int x, int y) {
   // Original implementation of foo
   return x+1;
}

void checkNoVariants();
void checkNoContext();
void checkDepend();

void declareVariant1()
{
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int y = 0;
  int output = 0;

  foo(x,y);

  #pragma omp dispatch
  output = foo(x,y);

  checkNoVariants();
  checkNoContext();
  checkDepend();
}

void checkNoVariants()
{
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int y = 0;
  int output = 0;

  #pragma omp dispatch novariants(cond_false)
  foo(x,y);

  #pragma omp dispatch novariants(cond_true)
  output = foo(x,y);
}

void checkNoContext()
{
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int y = 0;
  int output = 0;

  #pragma omp dispatch nocontext(cond_false)
  foo(x,y);

  #pragma omp dispatch nocontext(cond_true)
  output = foo(x,y);

}

void checkDepend()
{
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int y = 0;
  int output = 0;

  #pragma omp dispatch depend(out:x)
  output = foo(x,y);

  #pragma omp dispatch depend(out:x) depend(out:y)
  output = foo(x,y);

  #pragma omp dispatch depend(out:x) novariants(cond_false)
  output = foo(x,y);

  #pragma omp dispatch depend(out:x) nocontext(cond_false)
  foo(x,y);

  #pragma omp dispatch depend(out:x) novariants(cond_false) nocontext(cond_true)
  output = foo(x,y);
}

int bar_variant(int x) {
  return x+2;
}

int bar(int x) {
  return x+1;
}

void checkNoContext_withoutVariant();
void checkNoVariants_withoutVariant();

int without_declareVariant()
{
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int output = 0;

  bar(x);

  #pragma omp dispatch
  bar(x);

  checkNoVariants_withoutVariant();
  checkNoContext_withoutVariant();
  return 1;
}

void checkNoVariants_withoutVariant() {
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int output = 0;

  #pragma omp dispatch novariants(cond_true)
  output = bar(x);

  #pragma omp dispatch novariants(cond_false)
  bar(x);
}

void checkNoContext_withoutVariant() {
  int cond_false = 0, cond_true = 1;

  int x = 0;
  int output = 0;

  #pragma omp dispatch nocontext(cond_true)
  output = bar(x);

  #pragma omp dispatch nocontext(cond_false)
  bar(x);
}

// The dispatch region is emitted inline. Each 'novariants'/'nocontext' clause
// selects between the call to the base function or the variant chosen without
// the dispatch construct in context, and the call in the associated statement.

// CHECK-LABEL: define {{.+}}declareVariant1{{.+}}
// CHECK: call {{.+}}@_Z19foo_variant_allCondii
//
// #pragma omp dispatch
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z20foo_variant_dispatchii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK-NEXT: call {{.+}}checkNoVariants{{.+}}
// CHECK-NEXT: call {{.+}}checkNoContext{{.+}}
// CHECK-NEXT: call {{.+}}checkDepend{{.+}}
// CHECK-NEXT: ret void

// CHECK-LABEL: define {{.+}}checkNoVariants{{.+}}
//
// #pragma omp dispatch novariants(cond_false)
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: call {{.+}}@_Z3fooii
// CHECK: [[ELSE]]:
// CHECK: call {{.+}}@_Z20foo_variant_dispatchii
//
// #pragma omp dispatch novariants(cond_true)
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z3fooii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK: [[ELSE]]:
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z20foo_variant_dispatchii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK: ret void

// CHECK-LABEL: define {{.+}}checkNoContext{{.+}}
//
// #pragma omp dispatch nocontext(cond_false)
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: call {{.+}}@_Z19foo_variant_allCondii
// CHECK: [[ELSE]]:
// CHECK: call {{.+}}@_Z20foo_variant_dispatchii
//
// #pragma omp dispatch nocontext(cond_true)
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z19foo_variant_allCondii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK: [[ELSE]]:
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z20foo_variant_dispatchii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK: ret void

// CHECK-LABEL: define {{.+}}checkDepend{{.+}}
//
// #pragma omp dispatch depend(out:x)
// CHECK: call {{.+}}@__kmpc_omp_taskwait_deps_51({{.+}}, i32 1,
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z20foo_variant_dispatchii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
//
// #pragma omp dispatch depend(out:x) depend(out:y)
// CHECK: call {{.+}}@__kmpc_omp_taskwait_deps_51({{.+}}, i32 2,
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z20foo_variant_dispatchii
// CHECK-NEXT: store i32 [[CALL]], ptr %output
//
// #pragma omp dispatch depend(out:x) novariants(cond_false)
// CHECK: call {{.+}}@__kmpc_omp_taskwait_deps_51
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: call {{.+}}@_Z3fooii
// CHECK: [[ELSE]]:
// CHECK: call {{.+}}@_Z20foo_variant_dispatchii
//
// #pragma omp dispatch depend(out:x) nocontext(cond_false)
// CHECK: call {{.+}}@__kmpc_omp_taskwait_deps_51
// CHECK: br i1 %{{.+}}, label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK: call {{.+}}@_Z19foo_variant_allCondii
// CHECK: [[ELSE]]:
// CHECK: call {{.+}}@_Z20foo_variant_dispatchii
//
// #pragma omp dispatch depend(out:x) novariants(cond_false) nocontext(cond_true)
// The 'nocontext' test is nested in the 'novariants' else branch, and is on a
// different condition.
// CHECK: call {{.+}}@__kmpc_omp_taskwait_deps_51
// CHECK: [[NV:%.+]] = load i8, ptr [[NVCOND:%.capture_expr.+]],
// CHECK: [[NVB:%.+]] = icmp ne i8 [[NV]], 0
// CHECK: br i1 [[NVB]], label %[[NVTHEN:.+]], label %[[NVELSE:.+]]
// CHECK: [[NVTHEN]]:
// CHECK: call {{.+}}@_Z3fooii
// CHECK: [[NVELSE]]:
// CHECK-NOT: [[NVCOND]]
// CHECK: [[NC:%.+]] = load i8, ptr %.capture_expr.
// CHECK: [[NCB:%.+]] = icmp ne i8 [[NC]], 0
// CHECK: br i1 [[NCB]], label %[[NCTHEN:.+]], label %[[NCELSE:.+]]
// CHECK: [[NCTHEN]]:
// CHECK: call {{.+}}@_Z19foo_variant_allCondii
// CHECK: [[NCELSE]]:
// CHECK: call {{.+}}@_Z20foo_variant_dispatchii
// CHECK: ret void

// CHECK-LABEL: define {{.+}}without_declareVariant{{.+}}
// CHECK: call {{.+}}@_Z3bari
//
// #pragma omp dispatch
// CHECK: call {{.+}}@_Z3bari
// CHECK-NEXT: call {{.+}}checkNoVariants_withoutVariant{{.+}}
// CHECK-NEXT: call {{.+}}checkNoContext_withoutVariant{{.+}}
// CHECK-NEXT: ret

// Without a variant there is nothing for the clauses to select between.
// CHECK-LABEL: define {{.+}}checkNoVariants_withoutVariant{{.+}}
// CHECK-NOT: br i1
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z3bari
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK-NOT: br i1
// CHECK: call {{.+}}@_Z3bari
// CHECK-NEXT: ret void

// CHECK-LABEL: define {{.+}}checkNoContext_withoutVariant{{.+}}
// CHECK-NOT: br i1
// CHECK: [[CALL:%.+]] = call {{.+}}@_Z3bari
// CHECK-NEXT: store i32 [[CALL]], ptr %output
// CHECK-NOT: br i1
// CHECK: call {{.+}}@_Z3bari
// CHECK-NEXT: ret void
