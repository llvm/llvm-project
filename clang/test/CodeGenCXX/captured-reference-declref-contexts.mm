// RUN: %clang_cc1 -std=gnu++17 -fopenmp -fopenmp-version=45 -fblocks -fexceptions -fobjc-exceptions -triple x86_64-apple-darwin12 -disable-O0-optnone -emit-llvm %s -o - | FileCheck %s

/// Regression tests for `EmitDeclRefLValue`: a `DeclRefExpr` to a local reference
/// variable must not take the `NOUR_Constant` fast path through to the referee
/// when that name is implemented via capture storage (lambda field, OpenMP
/// outlined region, block literal, ObjC @finally capture, etc.).
///
/// Contexts below are intentionally distinct:
/// - **Lambda `[r]`**: by-copy capture holds a copy of the referenced object; the
///   body must not mutate `global` when the outer binding was `int &r = global`.
/// - **Lambda `[&r]`**: contrast — capture holds the address of `r`; mutation
///   still reaches `global`.
/// - **OpenMP `firstprivate`**: each thread gets private storage initialized from
///   the outer `r`; the outlined body must not store through `@global`.
/// - **Block**: Clang stores the *address* of the referee in the block literal; the
///   invoke must still load that pointer from the capture field. Using
///   `int *p = &global; int &r = *p` avoids a global block with no captures that
///   would not exercise this path.
/// - **`@finally`**: the body is a captured statement (`CR_ObjCAtFinally`). Use
///   `int *p = &global; int &r = *p` like the block case: plain `int &r = global`
///   can be lowered as a direct store to `@global` in the `@finally` block, which
///   does not exercise the capture path; indirection keeps the `DeclRef` for `r`
///   tied to capture lowering.

int global;

void f1() {
  int &r = global;
  r = 1;
  auto L = [r]() mutable { r = 99; };
  L();
}

void f2() {
  int &r = global;
  r = 2;
  auto L = [&r]() mutable { r = 88; };
  L();
}

void omp_firstprivate_ref_to_global(void) {
  int &r = global;
  r = 7;
#pragma omp parallel num_threads(1) default(none) firstprivate(r)
  {
    r = 8421;
  }
}

void block_ref_to_global(void) {
  int *p = &global;
  int &r = *p;
  r = 1;
  void (^b)(void) = ^{
    r = 8423;
  };
  b();
}

void finally_ref_to_global(void) {
  int *p = &global;
  int &r = *p;
  r = 7;
  @try {
  } @finally {
    r = 8424;
  }
}

// --- Lambda by-copy: must not store into @global inside the closure.
// CHECK-LABEL: define internal void @"{{_ZZ2f1vE.+clEv}}"
// CHECK-NOT: store i32 99, ptr @global
// CHECK: store i32 99

// --- Lambda by-reference: write still reaches the referee (value 88).
// CHECK-LABEL: define internal void @"{{_ZZ2f2vE.+clEv}}"
// CHECK: store i32 88

// --- OpenMP firstprivate: outlined region must not store into @global.
// CHECK-LABEL: define {{.*}}@{{.*}}omp_firstprivate_ref_to_global
// CHECK: define internal {{.*}}@{{.*}}.omp_outlined
// CHECK-NOT: store i32 8421, ptr @global

// --- Block: invoke must not fold to a direct store to @global.
// CHECK-LABEL: define {{.*}}@_Z{{.*}}block_ref_to_global
// CHECK: define internal {{.*}} @{{.*}}_block_invoke
// CHECK-NOT: store i32 8423, ptr @global
// CHECK: store i32 8423

// --- ObjC @finally: must not fold to a direct store to @global (see header).
// CHECK-LABEL: define {{.*}}@_Z{{.*}}finally_ref_to_global
// CHECK-NOT: store i32 8424, ptr @global
// CHECK: store i32 8424
