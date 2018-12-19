// Test CodeGen for Security Check Overflow Builtins.
// rdar://13421498

// RUN: %clang_cc1 -triple "i686-unknown-unknown"   -emit-llvm -x c %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple "x86_64-unknown-unknown" -emit-llvm -x c %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple "x86_64-mingw32"         -emit-llvm -x c %s -o - | FileCheck %s

extern unsigned UnsignedErrorCode;
extern unsigned long UnsignedLongErrorCode;
extern unsigned long long UnsignedLongLongErrorCode;
extern int IntErrorCode;
extern long LongErrorCode;
extern long long LongLongErrorCode;
void overflowed(void);

unsigned test_add_overflow_uint_uint_uint(unsigned x, unsigned y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_add_overflow_uint_uint_uint
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  unsigned r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

int test_add_overflow_int_int_int(int x, int y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_add_overflow_int_int_int
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  int r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

unsigned test_sub_overflow_uint_uint_uint(unsigned x, unsigned y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_sub_overflow_uint_uint_uint
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  unsigned r;
  if (__builtin_sub_overflow(x, y, &r))
    overflowed();
  return r;
}

int test_sub_overflow_int_int_int(int x, int y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_sub_overflow_int_int_int
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  int r;
  if (__builtin_sub_overflow(x, y, &r))
    overflowed();
  return r;
}

unsigned test_mul_overflow_uint_uint_uint(unsigned x, unsigned y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_mul_overflow_uint_uint_uint
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  unsigned r;
  if (__builtin_mul_overflow(x, y, &r))
    overflowed();
  return r;
}

int test_mul_overflow_int_int_int(int x, int y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_mul_overflow_int_int_int
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  int r;
  if (__builtin_mul_overflow(x, y, &r))
    overflowed();
  return r;
}

int test_add_overflow_uint_int_int(unsigned x, int y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_add_overflow_uint_int_int
  // CHECK: [[XE:%.+]] = zext i32 %{{.+}} to i33
  // CHECK: [[YE:%.+]] = sext i32 %{{.+}} to i33
  // CHECK: [[S:%.+]] = call { i33, i1 } @llvm.sadd.with.overflow.i33(i33 [[XE]], i33 [[YE]])
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i33, i1 } [[S]], 0
  // CHECK-DAG: [[C1:%.+]] = extractvalue { i33, i1 } [[S]], 1
  // CHECK: [[QT:%.+]] = trunc i33 [[Q]] to i32
  // CHECK: [[QTE:%.+]] = sext i32 [[QT]] to i33
  // CHECK: [[C2:%.+]] = icmp ne i33 [[Q]], [[QTE]]
  // CHECK: [[C3:%.+]] = or i1 [[C1]], [[C2]]
  // CHECK: store i32 [[QT]], i32*
  // CHECK: br i1 [[C3]]
  int r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

_Bool test_add_overflow_uint_uint_bool(unsigned x, unsigned y) {
  // CHECK-LABEL: define {{.*}} i1 @test_add_overflow_uint_uint_bool
  // CHECK-NOT: ext
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C1:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: [[QT:%.+]] = trunc i32 [[Q]] to i1
  // CHECK: [[QTE:%.+]] = zext i1 [[QT]] to i32
  // CHECK: [[C2:%.+]] = icmp ne i32 [[Q]], [[QTE]]
  // CHECK: [[C3:%.+]] = or i1 [[C1]], [[C2]]
  // CHECK: [[QT2:%.+]] = zext i1 [[QT]] to i8
  // CHECK: store i8 [[QT2]], i8*
  // CHECK: br i1 [[C3]]
  _Bool r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

unsigned test_add_overflow_bool_bool_uint(_Bool x, _Bool y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_add_overflow_bool_bool_uint
  // CHECK: [[XE:%.+]] = zext i1 %{{.+}} to i32
  // CHECK: [[YE:%.+]] = zext i1 %{{.+}} to i32
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[XE]], i32 [[YE]])
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: store i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  unsigned r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

_Bool test_add_overflow_bool_bool_bool(_Bool x, _Bool y) {
  // CHECK-LABEL: define {{.*}} i1 @test_add_overflow_bool_bool_bool
  // CHECK: [[S:%.+]] = call { i1, i1 } @llvm.uadd.with.overflow.i1(i1 %{{.+}}, i1 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i1, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i1, i1 } [[S]], 1
  // CHECK: [[QT2:%.+]] = zext i1 [[Q]] to i8
  // CHECK: store i8 [[QT2]], i8*
  // CHECK: br i1 [[C]]
  _Bool r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return r;
}

int test_add_overflow_volatile(int x, int y) {
  // CHECK-LABEL: define {{(dso_local )?}}i32 @test_add_overflow_volatile
  // CHECK: [[S:%.+]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  // CHECK-DAG: [[Q:%.+]] = extractvalue { i32, i1 } [[S]], 0
  // CHECK-DAG: [[C:%.+]] = extractvalue { i32, i1 } [[S]], 1
  // CHECK: store volatile i32 [[Q]], i32*
  // CHECK: br i1 [[C]]
  volatile int result;
  if (__builtin_add_overflow(x, y, &result))
    overflowed();
  return result;
}

unsigned test_uadd_overflow(unsigned x, unsigned y) {
// CHECK: @test_uadd_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_uadd_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_uaddl_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_uaddl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.uadd.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_uaddl_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_uaddll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_uaddll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_uaddll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

unsigned test_usub_overflow(unsigned x, unsigned y) {
// CHECK: @test_usub_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_usub_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_usubl_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_usubl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.usub.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_usubl_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_usubll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_usubll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_usubll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

unsigned test_umul_overflow(unsigned x, unsigned y) {
// CHECK: @test_umul_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  unsigned result;
  if (__builtin_umul_overflow(x, y, &result))
    return UnsignedErrorCode;
  return result;
}

unsigned long test_umull_overflow(unsigned long x, unsigned long y) {
// CHECK: @test_umull_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.umul.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  unsigned long result;
  if (__builtin_umull_overflow(x, y, &result))
    return UnsignedLongErrorCode;
  return result;
}

unsigned long long test_umulll_overflow(unsigned long long x, unsigned long long y) {
// CHECK: @test_umulll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  unsigned long long result;
  if (__builtin_umulll_overflow(x, y, &result))
    return UnsignedLongLongErrorCode;
  return result;
}

int test_sadd_overflow(int x, int y) {
// CHECK: @test_sadd_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_sadd_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_saddl_overflow(long x, long y) {
// CHECK: @test_saddl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.sadd.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_saddl_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_saddll_overflow(long long x, long long y) {
// CHECK: @test_saddll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_saddll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

int test_ssub_overflow(int x, int y) {
// CHECK: @test_ssub_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_ssub_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_ssubl_overflow(long x, long y) {
// CHECK: @test_ssubl_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.ssub.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_ssubl_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_ssubll_overflow(long long x, long long y) {
// CHECK: @test_ssubll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_ssubll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

int test_smul_overflow(int x, int y) {
// CHECK: @test_smul_overflow
// CHECK: %{{.+}} = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %{{.+}}, i32 %{{.+}})
  int result;
  if (__builtin_smul_overflow(x, y, &result))
    return IntErrorCode;
  return result;
}

long test_smull_overflow(long x, long y) {
// CHECK: @test_smull_overflow([[UL:i32|i64]] %x
// CHECK: %{{.+}} = call { [[UL]], i1 } @llvm.smul.with.overflow.[[UL]]([[UL]] %{{.+}}, [[UL]] %{{.+}})
  long result;
  if (__builtin_smull_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

long long test_smulll_overflow(long long x, long long y) {
// CHECK: @test_smulll_overflow
// CHECK: %{{.+}} = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %{{.+}}, i64 %{{.+}})
  long long result;
  if (__builtin_smulll_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

int test_mixed_sign_mul_overflow_sext_signed_op(int x, unsigned long long y) {
// CHECK: @test_mixed_sign_mul_overflow_sext_signed_op
// CHECK: [[SignedOp:%.*]] = sext i32 %0 to i64
// CHECK: [[IsNeg:%.*]] = icmp slt i64 [[SignedOp]], 0
  int result;
  if (__builtin_mul_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

int test_mixed_sign_mul_overflow_zext_unsigned_op(long long x, unsigned y) {
// CHECK: @test_mixed_sign_mul_overflow_zext_unsigned_op
// CHECK: [[UnsignedOp:%.*]] = zext i32 %1 to i64
// CHECK: [[IsNeg:%.*]] = icmp slt i64 %0, 0
// CHECK: @llvm.umul.with.overflow.i64({{.*}}, i64 [[UnsignedOp]])
  int result;
  if (__builtin_mul_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

int test_mixed_sign_mull_overflow(int x, unsigned y) {
// CHECK: @test_mixed_sign_mull_overflow
// CHECK:       [[IsNeg:%.*]] = icmp slt i32 [[Op1:%.*]], 0
// CHECK-NEXT:  [[Signed:%.*]] = sub i32 0, [[Op1]]
// CHECK-NEXT:  [[AbsSigned:%.*]] = select i1 [[IsNeg]], i32 [[Signed]], i32 [[Op1]]
// CHECK-NEXT:  call { i32, i1 } @llvm.umul.with.overflow.i32(i32 [[AbsSigned]], i32 %{{.*}})
// CHECK-NEXT:  [[UnsignedOFlow:%.*]] = extractvalue { i32, i1 } %{{.*}}, 1
// CHECK-NEXT:  [[UnsignedResult:%.*]] = extractvalue { i32, i1 } %{{.*}}, 0
// CHECK-NEXT:  [[IsNegZext:%.*]] = zext i1 [[IsNeg]] to i32
// CHECK-NEXT:  [[MaxResult:%.*]] = add i32 2147483647, [[IsNegZext]]
// CHECK-NEXT:  [[SignedOFlow:%.*]] = icmp ugt i32 [[UnsignedResult]], [[MaxResult]]
// CHECK-NEXT:  [[OFlow:%.*]] = or i1 [[UnsignedOFlow]], [[SignedOFlow]]
// CHECK-NEXT:  [[NegativeResult:%.*]] = sub i32 0, [[UnsignedResult]]
// CHECK-NEXT:  [[Result:%.*]] = select i1 [[IsNeg]], i32 [[NegativeResult]], i32 [[UnsignedResult]]
// CHECK-NEXT:  store i32 [[Result]], i32* %{{.*}}, align 4
// CHECK:       br i1 [[OFlow]]

  int result;
  if (__builtin_mul_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

int test_mixed_sign_mull_overflow_unsigned(int x, unsigned y) {
// CHECK: @test_mixed_sign_mull_overflow_unsigned
// CHECK:       [[IsNeg:%.*]] = icmp slt i32 [[Op1:%.*]], 0
// CHECK-NEXT:  [[Signed:%.*]] = sub i32 0, [[Op1]]
// CHECK-NEXT:  [[AbsSigned:%.*]] = select i1 [[IsNeg]], i32 [[Signed]], i32 [[Op1]]
// CHECK-NEXT:  call { i32, i1 } @llvm.umul.with.overflow.i32(i32 [[AbsSigned]], i32 %{{.*}})
// CHECK-NEXT:  [[UnsignedOFlow:%.*]] = extractvalue { i32, i1 } %{{.*}}, 1
// CHECK-NEXT:  [[UnsignedResult:%.*]] = extractvalue { i32, i1 } %{{.*}}, 0
// CHECK-NEXT:  [[NotNull:%.*]] = icmp ne i32 [[UnsignedResult]], 0
// CHECK-NEXT:  [[Underflow:%.*]] = and i1 [[IsNeg]], [[NotNull]]
// CHECK-NEXT:  [[OFlow:%.*]] = or i1 [[UnsignedOFlow]], [[Underflow]]
// CHECK-NEXT:  [[NegatedResult:%.*]] = sub i32 0, [[UnsignedResult]]
// CHECK-NEXT:  [[Result:%.*]] = select i1 [[IsNeg]], i32 [[NegatedResult]], i32 [[UnsignedResult]]
// CHECK-NEXT:  store i32 [[Result]], i32* %{{.*}}, align 4
// CHECK:       br i1 [[OFlow]]

  unsigned result;
  if (__builtin_mul_overflow(x, y, &result))
    return LongErrorCode;
  return result;
}

int test_mixed_sign_mull_overflow_swapped(int x, unsigned y) {
// CHECK: @test_mixed_sign_mull_overflow_swapped
// CHECK:  call { i32, i1 } @llvm.umul.with.overflow.i32
// CHECK:  add i32 2147483647
  int result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongErrorCode;
  return result;
}

long long test_mixed_sign_mulll_overflow(long long x, unsigned long long y) {
// CHECK: @test_mixed_sign_mulll_overflow
// CHECK:  call { i64, i1 } @llvm.umul.with.overflow.i64
// CHECK:  add i64 92233720368547
  long long result;
  if (__builtin_mul_overflow(x, y, &result))
    return LongLongErrorCode;
  return result;
}

long long test_mixed_sign_mulll_overflow_swapped(long long x, unsigned long long y) {
// CHECK: @test_mixed_sign_mulll_overflow_swapped
// CHECK:  call { i64, i1 } @llvm.umul.with.overflow.i64
// CHECK:  add i64 92233720368547
  long long result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongLongErrorCode;
  return result;
}

long long test_mixed_sign_mulll_overflow_trunc_signed(long long x, unsigned long long y) {
// CHECK: @test_mixed_sign_mulll_overflow_trunc_signed
// CHECK:  call { i64, i1 } @llvm.umul.with.overflow.i64
// CHECK:  add i64 2147483647
// CHECK:  trunc
// CHECK:  store
  int result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongLongErrorCode;
  return result;
}

long long test_mixed_sign_mulll_overflow_trunc_unsigned(long long x, unsigned long long y) {
// CHECK: @test_mixed_sign_mulll_overflow_trunc_unsigned
// CHECK:       call { i64, i1 } @llvm.umul.with.overflow.i64
// CHECK:       [[NON_ZERO:%.*]] = icmp ne i64 [[UNSIGNED_RESULT:%.*]], 0
// CHECK-NEXT:  [[UNDERFLOW:%.*]] = and i1 {{.*}}, [[NON_ZERO]]
// CHECK-NEXT:  [[OVERFLOW_PRE_TRUNC:%.*]] = or i1 {{.*}}, [[UNDERFLOW]]
// CHECK-NEXT:  [[TRUNC_OVERFLOW:%.*]] = icmp ugt i64 [[UNSIGNED_RESULT]], 4294967295
// CHECK-NEXT:  [[OVERFLOW:%.*]] = or i1 [[OVERFLOW_PRE_TRUNC]], [[TRUNC_OVERFLOW]]
// CHECK-NEXT:  [[NEGATED:%.*]] = sub i64 0, [[UNSIGNED_RESULT]]
// CHECK-NEXT:  [[RESULT:%.*]] = select i1 {{.*}}, i64 [[NEGATED]], i64 [[UNSIGNED_RESULT]]
// CHECK-NEXT:  trunc i64 [[RESULT]] to i32
// CHECK-NEXT:  store
  unsigned result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongLongErrorCode;
  return result;
}

long long test_mixed_sign_mul_overflow_extend_signed(int x, unsigned y) {
// CHECK: @test_mixed_sign_mul_overflow_extend_signed
// CHECK:  call { i64, i1 } @llvm.smul.with.overflow.i64
  long long result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongLongErrorCode;
  return result;
}

long long test_mixed_sign_mul_overflow_extend_unsigned(int x, unsigned y) {
// CHECK: @test_mixed_sign_mul_overflow_extend_unsigned
// CHECK:  call { i65, i1 } @llvm.smul.with.overflow.i65
  unsigned long long result;
  if (__builtin_mul_overflow(y, x, &result))
    return LongLongErrorCode;
  return result;
}
