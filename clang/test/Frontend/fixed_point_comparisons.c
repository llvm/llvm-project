// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNPADDED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,PADDED

// Fixed point against other fixed point
_Bool b_eq_true = 2.5hk == 2.5uhk;  // CHECK-DAG: @b_eq_true  = {{.*}}global i8 1, align 1
_Bool b_eq_false = 2.5hk == 2.4uhk; // CHECK-DAG: @b_eq_false = {{.*}}global i8 0, align 1

_Bool b_ne_true = 2.5hk != 2.4uhk;  // CHECK-DAG: @b_ne_true  = {{.*}}global i8 1, align 1
_Bool b_ne_false = 2.5hk != 2.5uhk; // CHECK-DAG: @b_ne_false = {{.*}}global i8 0, align 1

_Bool b_lt_true = 2.5hk < 2.75uhk; // CHECK-DAG: @b_lt_true  = {{.*}}global i8 1, align 1
_Bool b_lt_false = 2.5hk < 2.5uhk; // CHECK-DAG: @b_lt_false = {{.*}}global i8 0, align 1

_Bool b_le_true = 2.5hk <= 2.75uhk; // CHECK-DAG: @b_le_true  = {{.*}}global i8 1, align 1
_Bool b_le_true2 = 2.5hk <= 2.5uhk; // CHECK-DAG: @b_le_true2 = {{.*}}global i8 1, align 1
_Bool b_le_false = 2.5hk <= 2.4uhk; // CHECK-DAG: @b_le_false = {{.*}}global i8 0, align 1

_Bool b_gt_true = 2.75hk > 2.5uhk;   // CHECK-DAG: @b_gt_true  = {{.*}}global i8 1, align 1
_Bool b_gt_false = 2.75hk > 2.75uhk; // CHECK-DAG: @b_gt_false = {{.*}}global i8 0, align 1

_Bool b_ge_true = 2.75hk >= 2.5uhk;   // CHECK-DAG: @b_ge_true  = {{.*}}global i8 1, align 1
_Bool b_ge_true2 = 2.75hk >= 2.75uhk; // CHECK-DAG: @b_ge_true2 = {{.*}}global i8 1, align 1
_Bool b_ge_false = 2.5hk >= 2.75uhk;  // CHECK-DAG: @b_ge_false = {{.*}}global i8 0, align 1

// Fixed point against int
_Bool b_ieq_true = 2.0hk == 2;  // CHECK-DAG: @b_ieq_true  = {{.*}}global i8 1, align 1
_Bool b_ieq_false = 2.0hk == 3; // CHECK-DAG: @b_ieq_false = {{.*}}global i8 0, align 1

_Bool b_ine_true = 2.0hk != 3;  // CHECK-DAG: @b_ine_true  = {{.*}}global i8 1, align 1
_Bool b_ine_false = 2.0hk != 2; // CHECK-DAG: @b_ine_false = {{.*}}global i8 0, align 1

_Bool b_ilt_true = 2.0hk < 3;  // CHECK-DAG: @b_ilt_true  = {{.*}}global i8 1, align 1
_Bool b_ilt_false = 2.0hk < 2; // CHECK-DAG: @b_ilt_false = {{.*}}global i8 0, align 1

_Bool b_ile_true = 2.0hk <= 3;  // CHECK-DAG: @b_ile_true  = {{.*}}global i8 1, align 1
_Bool b_ile_true2 = 2.0hk <= 2; // CHECK-DAG: @b_ile_true2 = {{.*}}global i8 1, align 1
_Bool b_ile_false = 2.0hk <= 1; // CHECK-DAG: @b_ile_false = {{.*}}global i8 0, align 1

_Bool b_igt_true = 2.0hk > 1;  // CHECK-DAG: @b_igt_true  = {{.*}}global i8 1, align 1
_Bool b_igt_false = 2.0hk > 2; // CHECK-DAG: @b_igt_false = {{.*}}global i8 0, align 1

_Bool b_ige_true = 2.0hk >= 1;  // CHECK-DAG: @b_ige_true  = {{.*}}global i8 1, align 1
_Bool b_ige_true2 = 2.0hk >= 2; // CHECK-DAG: @b_ige_true2 = {{.*}}global i8 1, align 1
_Bool b_ige_false = 2.0hk >= 3; // CHECK-DAG: @b_ige_false = {{.*}}global i8 0, align 1

// Different signage
// Since we can have different precisions, non powers of 2 fractions may have
// different actual values when being compared.
_Bool b_sne_true = 2.6hk != 2.6uhk;
// UNPADDED-DAG:   @b_sne_true = {{.*}}global i8 1, align 1
// PADDED-DAG: @b_sne_true = {{.*}}global i8 0, align 1

_Bool b_seq_true = 2.0hk == 2u;  // CHECK-DAG: @b_seq_true  = {{.*}}global i8 1, align 1
_Bool b_seq_true2 = 2.0uhk == 2; // CHECK-DAG: @b_seq_true2 = {{.*}}global i8 1, align 1

void TestComparisons() {
  short _Accum sa;
  _Accum a;
  unsigned short _Accum usa;
  unsigned _Accum ua;

  // Each of these should be a fixed point conversion followed by the actual
  // comparison operation.
  sa == a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp eq i32 [[UPSCALE_A]], [[A2]]

  sa != a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp ne i32 [[UPSCALE_A]], [[A2]]

  sa > a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp sgt i32 [[UPSCALE_A]], [[A2]]

  sa >= a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp sge i32 [[UPSCALE_A]], [[A2]]

  sa < a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp slt i32 [[UPSCALE_A]], [[A2]]

  sa <= a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp sle i32 [[UPSCALE_A]], [[A2]]

  usa > ua;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp ugt i32 [[UPSCALE_A]], [[A2]]

  usa >= ua;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp uge i32 [[UPSCALE_A]], [[A2]]

  usa < ua;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp ult i32 [[UPSCALE_A]], [[A2]]

  usa <= ua;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %ua, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp ule i32 [[UPSCALE_A]], [[A2]]
}

void TestIntComparisons() {
  short _Accum sa;
  unsigned short _Accum usa;

  int i;
  unsigned int ui;
  _Bool b;
  char c;
  short s;
  enum E {
    A = 2
  } e;

  // These comparisons shouldn't be that different from comparing against fixed
  // point types with other fixed point types.
  sa == i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_I]]

  sa != i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp ne i39 [[RESIZE_A]], [[UPSCALE_I]]

  sa > i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp sgt i39 [[RESIZE_A]], [[UPSCALE_I]]

  sa >= i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp sge i39 [[RESIZE_A]], [[UPSCALE_I]]

  sa < i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp slt i39 [[RESIZE_A]], [[UPSCALE_I]]

  sa <= i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp sle i39 [[RESIZE_A]], [[UPSCALE_I]]

  usa > ui;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp ugt i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp ugt i39 [[RESIZE_A]], [[UPSCALE_I]]

  usa >= ui;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp uge i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp uge i39 [[RESIZE_A]], [[UPSCALE_I]]

  usa < ui;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp ult i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp ult i39 [[RESIZE_A]], [[UPSCALE_I]]

  usa <= ui;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp ule i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp ule i39 [[RESIZE_A]], [[UPSCALE_I]]

  // Allow for comparisons with other int like types. These are no different
  // from comparing to an int other than varying sizes. The integer types are
  // still converted to ints or unsigned ints from UsualUnaryConversions().
  sa == b;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[B:%[0-9]+]] = load i8, i8* %b, align 1
  // CHECK-NEXT: %tobool = trunc i8 [[B]] to i1
  // CHECK-NEXT: [[CONV_B:%[a-z0-9]+]] = zext i1 %tobool to i32
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_B:%[a-z0-9]+]] = sext i32 [[CONV_B]] to i39
  // CHECK-NEXT: [[UPSCALE_B:%[a-z0-9]+]] = shl i39 [[RESIZE_B]], 7
  // CHECK-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_B]]

  sa == c;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[C:%[0-9]+]] = load i8, i8* %c, align 1
  // CHECK-NEXT: [[CONV_C:%[a-z0-9]+]] = sext i8 [[C]] to i32
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_C:%[a-z0-9]+]] = sext i32 [[CONV_C]] to i39
  // CHECK-NEXT: [[UPSCALE_C:%[a-z0-9]+]] = shl i39 [[RESIZE_C]], 7
  // CHECK-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_C]]

  sa == s;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[S:%[0-9]+]] = load i16, i16* %s, align 2
  // CHECK-NEXT: [[CONV_S:%[a-z0-9]+]] = sext i16 [[S]] to i32
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i39
  // CHECK-NEXT: [[RESIZE_S:%[a-z0-9]+]] = sext i32 [[CONV_S]] to i39
  // CHECK-NEXT: [[UPSCALE_S:%[a-z0-9]+]] = shl i39 [[RESIZE_S]], 7
  // CHECK-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_S]]

  // An enum value is IntegralCast to an unsigned int.
  usa == e;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %e, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp eq i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_I]]
}

void TestComparisonSignage() {
  short _Accum sa;
  unsigned short _Accum usa;
  int i;
  unsigned int ui;

  // Signed vs unsigned fixed point comparison
  sa == usa;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i16, i16* %usa, align 2
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i17
  // UNPADDED-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i17 [[RESIZE_A]], 1
  // UNPADDED-NEXT: [[RESIZE_A2:%[a-z0-9]+]] = zext i16 [[A2]] to i17
  // UNPADDED-NEXT: {{.*}} = icmp eq i17 [[UPSCALE_A]], [[RESIZE_A2]]
  // PADDED-NEXT: {{.*}} = icmp eq i16 [[A]], [[A2]]

  // Signed int vs unsigned fixed point
  sa == ui;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %ui, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i40
  // CHECK-NEXT: [[RESIZE_I:%[a-z0-9]+]] = zext i32 [[I]] to i40
  // CHECK-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 7
  // CHECK-NEXT: {{.*}} = icmp eq i40 [[RESIZE_A]], [[UPSCALE_I]]

  // Signed fixed point vs unsigned int
  usa == i;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %usa, align 2
  // CHECK-NEXT: [[I:%[0-9]+]] = load i32, i32* %i, align 4
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i40
  // UNPADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i40
  // UNPADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i40 [[RESIZE_I]], 8
  // UNPADDED-NEXT: {{.*}} = icmp eq i40 [[RESIZE_A]], [[UPSCALE_I]]
  // PADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = zext i16 [[A]] to i39
  // PADDED-NEXT: [[RESIZE_I:%[a-z0-9]+]] = sext i32 [[I]] to i39
  // PADDED-NEXT: [[UPSCALE_I:%[a-z0-9]+]] = shl i39 [[RESIZE_I]], 7
  // PADDED-NEXT: {{.*}} = icmp eq i39 [[RESIZE_A]], [[UPSCALE_I]]
}

void TestSaturationComparisons() {
  short _Accum sa;
  _Accum a;
  _Sat short _Accum sat_sa;
  _Sat _Accum sat_a;
  _Sat unsigned short _Accum sat_usa;

  // These are effectively the same as conversions with their non-saturating
  // counterparts since when comparing, we convert both operands to a common
  // type that should be able to hold both values.
  sat_sa == sat_a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sat_sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %sat_a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp eq i32 [[UPSCALE_A]], [[A2]]

  sat_sa == a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sat_sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: {{.*}} = icmp eq i32 [[UPSCALE_A]], [[A2]]

  sat_sa == sat_usa;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sat_sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i16, i16* %sat_usa, align 2
  // UNPADDED-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i17
  // UNPADDED-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i17 [[RESIZE_A]], 1
  // UNPADDED-NEXT: [[RESIZE_A2:%[a-z0-9]+]] = zext i16 [[A2]] to i17
  // UNPADDED-NEXT: {{.*}} = icmp eq i17 [[UPSCALE_A]], [[RESIZE_A2]]
  // PADDED-NEXT: {{.*}} = icmp eq i16 [[A]], [[A2]]
}

void StoreBooleanResult() {
  short _Accum sa;
  _Accum a;
  int res;

  // Check that the result can properly be stored as an int.
  res = sa == a;
  // CHECK:      [[A:%[0-9]+]] = load i16, i16* %sa, align 2
  // CHECK-NEXT: [[A2:%[0-9]+]] = load i32, i32* %a, align 4
  // CHECK-NEXT: [[RESIZE_A:%[a-z0-9]+]] = sext i16 [[A]] to i32
  // CHECK-NEXT: [[UPSCALE_A:%[a-z0-9]+]] = shl i32 [[RESIZE_A]], 8
  // CHECK-NEXT: [[RES:%[0-9]+]] = icmp eq i32 [[UPSCALE_A]], [[A2]]
  // CHECK-NEXT: %conv = zext i1 [[RES]] to i32
  // CHECK-NEXT: store i32 %conv, i32* %res, align 4
}
