// Check the various ways in which the three classes of values
// (scalar, complex, aggregate) interact with parameter passing
// (function entry, function return, call argument, call result).
//
// We also check _Bool and empty structures, as these can have annoying
// corner cases.

// RUN: %clang_cc1 %s -triple i386-unknown-unknown -O3 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-unknown -O3 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc-unknown-unknown -O3 -emit-llvm -o - | FileCheck %s
// CHECK-NOT: @g0

typedef _Bool BoolTy;
typedef int ScalarTy;
typedef _Complex int ComplexTy;
typedef struct { int a, b, c; } AggrTy;
typedef struct { int a[0]; } EmptyTy;

static int result;

static BoolTy bool_id(BoolTy a) { return a; }
static AggrTy aggr_id(AggrTy a) { return a; }
static EmptyTy empty_id(EmptyTy a) { return a; }
static ScalarTy scalar_id(ScalarTy a) { return a; }
static ComplexTy complex_id(ComplexTy a) { return a; }

static void bool_mul(BoolTy a) { result *= a; }

static void aggr_mul(AggrTy a) { result *= a.a * a.b * a.c; }

static void empty_mul(EmptyTy a) { result *= 53; }

static void scalar_mul(ScalarTy a) { result *= a; }

static void complex_mul(ComplexTy a) { result *= __real a * __imag a; }

extern void g0(void);

void f0(void) {
  result = 1;
  
  bool_mul(bool_id(1));
  aggr_mul(aggr_id((AggrTy) { 2, 3, 5}));
  empty_mul(empty_id((EmptyTy) {}));
  scalar_mul(scalar_id(7));
  complex_mul(complex_id(11 + 13i));
  
  // This call should be eliminated.
  if (result != 2 * 3 * 5 * 7 * 11 * 13 * 53)
    g0();
}

