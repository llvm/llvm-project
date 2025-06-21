// expected-no-diagnostics
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ %s -emit-llvm -o - | FileCheck %s
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

// CHECK-LABEL: define {{.+}}declareVariant{{.+}}
// CHECK-LABEL: entry:
//
// #pragma omp dispatch
// CHECK: call {{.+}}foo_variant_allCond{{.+}}
// CHECK: call {{.+}}captured_stmt{{.+}}
// CHECK-NEXT: call {{.+}}checkNoVariants{{.+}}
// CHECK-NEXT: call {{.+}}checkNoContext{{.+}}
// CHECK-NEXT: call {{.+}}checkDepend{{.+}}
// CHECK-NEXT: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: {{.+}}checkNoVariants{{.+}}
// CHECK-LABEL: entry:
//  #pragma omp dispatch novariants(cond_false)
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.1{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.2{{.+}}
// CHECK-LABEL: if.end{{.+}}
//
//  #pragma omp dispatch novariants(cond_true)
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.3{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.4{{.+}}
// CHECK-LABEL: if.end{{.+}}
//
// CHECK-LABEL: {{.+}}checkNoContext{{.+}}
// CHECK-LABEL: entry:
//
//  #pragma omp dispatch nocontext(cond_false)
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.5{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.6{{.+}}
// CHECK-LABEL: if.end{{.+}}
//
//  #pragma omp dispatch nocontext(cond_true)
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.7{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.8{{.+}}
// CHECK-LABEL: if.end{{.+}}
//
// CHECK-LABEL: {{.+}}checkDepend{{.+}}
// CHECK-LABEL: entry:
//
//  #pragma omp dispatch depend(out:x)
// CHECK: call {{.+}}kmpc_omp_taskwait_deps{{.+}}
//
//  #pragma omp dispatch depend(out:x) depend(out:y)
// CHECK: call {{.+}}kmpc_omp_taskwait_deps{{.+}}
//
//  #pragma omp dispatch depend(out:x) novariants(cond_false)
// CHECK: call {{.+}}kmpc_omp_taskwait_deps{{.+}}
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.9{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.10{{.+}}
// CHECK-LABEL: if.end{{.+}}
//
//  #pragma omp dispatch depend(out:x) nocontext(cond_false)
// CHECK: call {{.+}}kmpc_omp_taskwait_deps{{.+}}
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.11{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.12{{.+}}
// CHECK-LABEL: if.end{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.1{{.+}}
// CHECK: call {{.+}}foo{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.2{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.3{{.+}}
// CHECK: call {{.+}}foo{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.4{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.5{{.+}}
// CHECK: call {{.+}}foo_variant_allCond{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.6{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.7{{.+}}
// CHECK: call {{.+}}foo_variant_allCond{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.8{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.9{{.+}}
// CHECK: call {{.+}}foo{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.10{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.11{{.+}}
// CHECK: call {{.+}}foo_variant_allCond{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.12{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.13{{.+}}
// CHECK: call {{.+}}foo{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.14{{.+}}
// CHECK: call {{.+}}foo_variant_allCond{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}__captured_stmt.15{{.+}}
// CHECK: call {{.+}}foo_variant_dispatch{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}bar_variant{{.+}}
// CHECK-LABEL: entry:
// CHECK: ret{{.+}}
//
// CHECK-LABEL: define {{.+}}bar{{.+}}
// CHECK-LABEL: entry:
// CHECK: ret{{.+}}

// CHECK-LABEL: define {{.+}}without_declareVariant{{.+}}
// CHECK-LABEL: entry:
// CHECK: call {{.+}}bar{{.+}}
// CHECK: call {{.+}}captured_stmt.16{{.+}}
// CHECK-NEXT: call {{.+}}checkNoVariants_withoutVariant{{.+}}
// CHECK-NEXT: call {{.+}}checkNoContext_withoutVariant{{.+}}
// CHECK-NEXT: ret{{.+}}
//
//  #pragma omp dispatch
// CHECK-LABEL: define {{.+}}__captured_stmt.16{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
//
// CHECK-LABEL: define {{.+}}checkNoVariants_withoutVariant{{.+}}
// CHECK-LABEL: entry:
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.17{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.18{{.+}}
// CHECK-LABEL: if.end{{.+}}
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.19{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.20{{.+}}
// CHECK-LABEL: if.end{{.+}}
// CHECK: ret{{.+}}
//
// CHECK-LABEL: define {{.+}}checkNoContext_withoutVariant{{.+}}
// CHECK-LABEL: entry:
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.21{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.22{{.+}}
// CHECK-LABEL: if.end{{.+}}
// CHECK-LABEL: if.then{{.+}}
// CHECK: call {{.+}}captured_stmt.23{{.+}}
// CHECK-LABEL: if.else{{.+}}
// CHECK: call {{.+}}captured_stmt.24{{.+}}
// CHECK-LABEL: if.end{{.+}}
// CHECK: ret{{.+}} 
//
// #pragma omp dispatch novariants(cond_true)
// CHECK-LABEL: define {{.+}}__captured_stmt.17{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
// CHECK-LABEL: define {{.+}}__captured_stmt.18{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
//
// #pragma omp dispatch novariants(cond_false)
// CHECK-LABEL: define {{.+}}__captured_stmt.19{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
// CHECK-LABEL: define {{.+}}__captured_stmt.20{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
//
// #pragma omp dispatch nocontext(cond_true)
// CHECK-LABEL: define {{.+}}__captured_stmt.21{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
// CHECK-LABEL: define {{.+}}__captured_stmt.22{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
//
// #pragma omp dispatch nocontext(cond_false)
// CHECK-LABEL: define {{.+}}__captured_stmt.23{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
// CHECK-LABEL: define {{.+}}__captured_stmt.24{{.+}}
// CHECK: call {{.+}}bar{{.+}}
// CHECK: ret void
