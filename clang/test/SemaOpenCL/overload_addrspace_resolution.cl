// RUN: %clang_cc1 -cl-std=CL2.0 -emit-llvm -o - -triple x86_64-unknown-unknown %s | FileCheck %s

void __attribute__((overloadable)) foo(global int *a, global int *b);
void __attribute__((overloadable)) foo(generic int *a, generic int *b);
void __attribute__((overloadable)) bar(generic int *global *a, generic int *global *b);
void __attribute__((overloadable)) bar(generic int *generic *a, generic int *generic *b);

void kernel ker() {
  global int *a;
  global int *b;
  generic int *c;
  local int *d;
  generic int *generic *gengen;
  generic int *local *genloc;
  generic int *global *genglob;
  // CHECK: call void @_Z3fooPU8CLglobaliS0_(i32* undef, i32* undef)
  foo(a, b);
  // CHECK: call void @_Z3fooPU9CLgenericiS0_(i32* undef, i32* undef)
  foo(b, c);
  // CHECK: call void @_Z3fooPU9CLgenericiS0_(i32* undef, i32* undef)
  foo(a, d);

  // CHECK: call void @_Z3barPU9CLgenericPU9CLgenericiS2_(i32** undef, i32** undef)
  bar(gengen, genloc);
  // CHECK: call void @_Z3barPU9CLgenericPU9CLgenericiS2_(i32** undef, i32** undef)
  bar(gengen, genglob);
  // CHECK: call void @_Z3barPU8CLglobalPU9CLgenericiS2_(i32** undef, i32** undef)
  bar(genglob, genglob);
}
