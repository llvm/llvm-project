// OpenCL C v3.0 s6.5.1: integer division and remainder by zero, and signed
// division overflow (INT_MIN / -1), result in an unspecified value. By default
// for OpenCL C and C++ for OpenCL, integer / and % that may hit one of these
// cases are emitted as a frozen llvm.masked.{s,u}{div,rem} whose mask is false
// exactly for those operands. SPIR and SPIR-V get the guarded form instead.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,ON
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=clc++2021 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,ON
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -fdefined-int-divrem -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,ON
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,GUARD
// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,GUARD
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,GUARD
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,GUARD
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,OFF
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -disable-llvm-passes -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,OFF

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned short ushort;
typedef int int4 __attribute__((ext_vector_type(4)));
typedef uint uint4 __attribute__((ext_vector_type(4)));

// Check the divisor via select before the dividend so a poison dividend lane
// cannot poison the mask when the divisor is not -1.
// CHECK-LABEL: @{{.*}}f_sdiv
// ON:          [[NZ:%.*]] = icmp ne i32 [[B:%.*]], 0
// ON-NEXT:     [[NMIN:%.*]] = icmp ne i32 [[A:%.*]], -2147483648
// ON-NEXT:     [[NM1:%.*]] = icmp ne i32 [[B]], -1
// ON-NEXT:     [[NOV:%.*]] = select i1 [[NM1]], i1 true, i1 [[NMIN]]
// ON-NEXT:     [[SAFE:%.*]] = select i1 [[NZ]], i1 [[NOV]], i1 false
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 [[A]], i32 [[B]], i1 [[SAFE]])
// ON-NEXT:     [[R:%.*]] = freeze i32 [[Q]]
// ON-NEXT:     ret i32 [[R]]
// GUARD:       [[NZ:%.*]] = icmp ne i32 [[B:%.*]], 0
// GUARD-NEXT:  [[NMIN:%.*]] = icmp ne i32 [[A:%.*]], -2147483648
// GUARD-NEXT:  [[NM1:%.*]] = icmp ne i32 [[B]], -1
// GUARD-NEXT:  [[NOV:%.*]] = select i1 [[NM1]], i1 true, i1 [[NMIN]]
// GUARD-NEXT:  [[SAFE:%.*]] = select i1 [[NZ]], i1 [[NOV]], i1 false
// GUARD-NEXT:  [[D:%.*]] = select i1 [[SAFE]], i32 [[B]], i32 1
// GUARD-NEXT:  [[R:%.*]] = sdiv i32 [[A]], [[D]]
// GUARD-NEXT:  ret i32 [[R]]
// OFF:         [[R:%.*]] = sdiv i32 %a, %b
// OFF-NEXT:    ret i32 [[R]]
int f_sdiv(int a, int b) { return a / b; }

// CHECK-LABEL: @{{.*}}f_srem
// ON:          [[SAFE:%.*]] = select i1 {{.*}}, i1 {{.*}}, i1 false
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.srem.i32(i32 %a, i32 %b, i1 [[SAFE]])
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[SAFE:%.*]] = select i1 {{.*}}, i1 {{.*}}, i1 false
// GUARD-NEXT:  [[D:%.*]] = select i1 [[SAFE]], i32 %b, i32 1
// GUARD-NEXT:  srem i32 %a, [[D]]
// OFF:         srem i32 %a, %b
int f_srem(int a, int b) { return a % b; }

// Unsigned operations only need a non-zero divisor.
// CHECK-LABEL: @{{.*}}f_udiv
// ON:          [[NZ:%.*]] = icmp ne i32 %b, 0
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.udiv.i32(i32 %a, i32 %b, i1 [[NZ]])
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[NZ:%.*]] = icmp ne i32 %b, 0
// GUARD-NEXT:  [[D:%.*]] = select i1 [[NZ]], i32 %b, i32 1
// GUARD-NEXT:  udiv i32 %a, [[D]]
// OFF:         udiv i32 %a, %b
uint f_udiv(uint a, uint b) { return a / b; }

// CHECK-LABEL: @{{.*}}f_urem
// ON:          [[NZ:%.*]] = icmp ne i64 %b, 0
// ON-NEXT:     [[Q:%.*]] = call i64 @llvm.masked.urem.i64(i64 %a, i64 %b, i1 [[NZ]])
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       [[NZ:%.*]] = icmp ne i64 %b, 0
// GUARD-NEXT:  [[D:%.*]] = select i1 [[NZ]], i64 %b, i64 1
// GUARD-NEXT:  urem i64 %a, [[D]]
// OFF:         urem i64 %a, %b
ulong f_urem(ulong a, ulong b) { return a % b; }

// CHECK-LABEL: @{{.*}}f_ldiv
// ON:          icmp ne i64 %a, -9223372036854775808
// ON:          [[Q:%.*]] = call i64 @llvm.masked.sdiv.i64(i64 %a, i64 %b, i1
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i64 %b, i64 1
// GUARD-NEXT:  sdiv i64 %a, [[D]]
// OFF:         sdiv i64 %a, %b
long f_ldiv(long a, long b) { return a / b; }

// char and short are promoted to int first.
// CHECK-LABEL: @{{.*}}f_char
// CHECK:       [[A:%.*]] = sext i8 %a to i32
// CHECK:       [[B:%.*]] = sext i8 %b to i32
// ON:          [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 [[A]], i32 [[B]], i1
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i32 [[B]], i32 1
// GUARD-NEXT:  sdiv i32 [[A]], [[D]]
// OFF:         sdiv i32 [[A]], [[B]]
char f_char(char a, char b) { return a / b; }

// CHECK-LABEL: @{{.*}}f_ushort
// CHECK:       [[A:%.*]] = zext i16 %a to i32
// CHECK:       [[B:%.*]] = zext i16 %b to i32
// ON:          [[Q:%.*]] = call i32 @llvm.masked.srem.i32(i32 [[A]], i32 [[B]], i1
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i32 [[B]], i32 1
// GUARD-NEXT:  srem i32 [[A]], [[D]]
// OFF:         srem i32 [[A]], [[B]]
ushort f_ushort(ushort a, ushort b) { return a % b; }

// Vectors are checked lane by lane.
// CHECK-LABEL: @{{.*}}f_vdiv
// ON:          [[NZ:%.*]] = icmp ne <4 x i32> [[B:%.*]], zeroinitializer
// ON-NEXT:     [[NMIN:%.*]] = icmp ne <4 x i32> [[A:%.*]], splat (i32 -2147483648)
// ON-NEXT:     [[NM1:%.*]] = icmp ne <4 x i32> [[B]], splat (i32 -1)
// ON-NEXT:     [[NOV:%.*]] = select <4 x i1> [[NM1]], <4 x i1> splat (i1 true), <4 x i1> [[NMIN]]
// ON-NEXT:     [[SAFE:%.*]] = select <4 x i1> [[NZ]], <4 x i1> [[NOV]], <4 x i1> zeroinitializer
// ON-NEXT:     [[Q:%.*]] = call <4 x i32> @llvm.masked.sdiv.v4i32(<4 x i32> [[A]], <4 x i32> [[B]], <4 x i1> [[SAFE]])
// ON-NEXT:     freeze <4 x i32> [[Q]]
// GUARD:       [[SAFE:%.*]] = select <4 x i1> {{.*}}, <4 x i1> {{.*}}, <4 x i1> zeroinitializer
// GUARD-NEXT:  [[D:%.*]] = select <4 x i1> [[SAFE]], <4 x i32> [[B:%.*]], <4 x i32> splat (i32 1)
// GUARD-NEXT:  sdiv <4 x i32> [[A:%.*]], [[D]]
// OFF:         sdiv <4 x i32>
int4 f_vdiv(int4 a, int4 b) { return a / b; }

// CHECK-LABEL: @{{.*}}f_vurem
// ON:          [[NZ:%.*]] = icmp ne <4 x i32> [[B:%.*]], zeroinitializer
// ON-NEXT:     [[Q:%.*]] = call <4 x i32> @llvm.masked.urem.v4i32(<4 x i32> [[A:%.*]], <4 x i32> [[B]], <4 x i1> [[NZ]])
// ON-NEXT:     freeze <4 x i32> [[Q]]
// GUARD:       [[NZ:%.*]] = icmp ne <4 x i32> [[B:%.*]], zeroinitializer
// GUARD-NEXT:  [[D:%.*]] = select <4 x i1> [[NZ]], <4 x i32> [[B]], <4 x i32> splat (i32 1)
// GUARD-NEXT:  urem <4 x i32> [[A:%.*]], [[D]]
// OFF:         urem <4 x i32>
uint4 f_vurem(uint4 a, uint4 b) { return a % b; }

// A scalar divisor is splat first.
// CHECK-LABEL: @{{.*}}f_vdiv_scalar
// CHECK:       [[SPLAT:%.*]] = shufflevector <4 x i32>
// ON:          icmp ne <4 x i32> [[SPLAT]], zeroinitializer
// ON:          [[Q:%.*]] = call <4 x i32> @llvm.masked.sdiv.v4i32(<4 x i32> %a, <4 x i32> [[SPLAT]], <4 x i1>
// ON-NEXT:     freeze <4 x i32> [[Q]]
// GUARD:       select <4 x i1> {{.*}}, <4 x i32> [[SPLAT]], <4 x i32> splat (i32 1)
// OFF:         sdiv <4 x i32> %a, [[SPLAT]]
int4 f_vdiv_scalar(int4 a, int b) { return a / b; }

// Compound assignments go through the same path.
// CHECK-LABEL: @{{.*}}f_divassign
// ON:          [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 %a, i32 %b, i1
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i32 %b, i32 1
// GUARD-NEXT:  sdiv i32 %a, [[D]]
// OFF:         sdiv i32 %a, %b
int f_divassign(int a, int b) {
  a /= b;
  return a;
}

// CHECK-LABEL: @{{.*}}f_remassign
// ON:          [[Q:%.*]] = call <4 x i32> @llvm.masked.urem.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i1>
// ON-NEXT:     freeze <4 x i32> [[Q]]
// GUARD:       [[D:%.*]] = select <4 x i1> {{.*}}, <4 x i32> %b, <4 x i32> splat (i32 1)
// GUARD-NEXT:  urem <4 x i32> %a, [[D]]
// OFF:         urem <4 x i32> %a, %b
uint4 f_remassign(uint4 a, uint4 b) {
  a %= b;
  return a;
}

// A constant divisor that rules out both cases keeps the plain instruction.
// CHECK-LABEL: @{{.*}}f_div_const
// CHECK-NOT:   select
// CHECK:       sdiv i32 %a, 4
// CHECK-NEXT:  ret
int f_div_const(int a) { return a / 4; }

// A vector literal is emitted through a temporary, so the divisor is not a
// constant yet when the operation is emitted.
// CHECK-LABEL: @{{.*}}f_vrem_literal
// ON:          [[Q:%.*]] = call <4 x i32> @llvm.masked.srem.v4i32(<4 x i32> %a, <4 x i32> <i32 1, i32 -2, i32 3, i32 -4>, <4 x i1>
// ON-NEXT:     freeze <4 x i32> [[Q]]
// GUARD:       [[D:%.*]] = select <4 x i1> {{.*}}, <4 x i32> <i32 1, i32 -2, i32 3, i32 -4>, <4 x i32> splat (i32 1)
// GUARD-NEXT:  srem <4 x i32> %a, [[D]]
// OFF:         srem <4 x i32> %a, <i32 1, i32 -2, i32 3, i32 -4>
int4 f_vrem_literal(int4 a) { return a % (int4)(1, -2, 3, -4); }

// CHECK-LABEL: @{{.*}}f_vdiv_splat
// CHECK-NOT:   select
// CHECK:       sdiv <4 x i32> %a, splat (i32 4)
// CHECK-NEXT:  ret
int4 f_vdiv_splat(int4 a) { return a / 4; }

// For unsigned operations, all-ones is a safe divisor.
// CHECK-LABEL: @{{.*}}f_udiv_allones
// CHECK-NOT:   select
// CHECK:       udiv i32 %a, -1
// CHECK-NEXT:  ret
uint f_udiv_allones(uint a) { return a / 0xffffffffu; }

// A constant dividend other than INT_MIN rules out overflow, but not a zero
// divisor.
// CHECK-LABEL: @{{.*}}f_div_const_dividend
// ON:          [[NZ:%.*]] = icmp ne i32 %b, 0
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 7, i32 %b, i1 [[NZ]])
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[NZ:%.*]] = icmp ne i32 %b, 0
// GUARD-NEXT:  [[D:%.*]] = select i1 [[NZ]], i32 %b, i32 1
// GUARD-NEXT:  sdiv i32 7, [[D]]
// OFF:         sdiv i32 7, %b
int f_div_const_dividend(int b) { return 7 / b; }

// A constant INT_MIN dividend only overflows when the divisor is -1.
// CHECK-LABEL: @{{.*}}f_div_min_dividend
// ON:          [[NZ:%.*]] = icmp ne i32 %b, 0
// ON-NEXT:     [[NM1:%.*]] = icmp ne i32 %b, -1
// ON-NEXT:     [[SAFE:%.*]] = select i1 [[NZ]], i1 [[NM1]], i1 false
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 -2147483648, i32 %b, i1 [[SAFE]])
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[NZ:%.*]] = icmp ne i32 %b, 0
// GUARD-NEXT:  [[NM1:%.*]] = icmp ne i32 %b, -1
// GUARD-NEXT:  [[SAFE:%.*]] = select i1 [[NZ]], i1 [[NM1]], i1 false
// GUARD-NEXT:  [[D:%.*]] = select i1 [[SAFE]], i32 %b, i32 1
// GUARD-NEXT:  sdiv i32 -2147483648, [[D]]
// OFF:         sdiv i32 -2147483648, %b
int f_div_min_dividend(int b) { return (-2147483647 - 1) / b; }

// A constant divisor of -1 can overflow, and with a dividend other than
// INT_MIN it cannot.
// CHECK-LABEL: @{{.*}}f_div_minus_one
// ON:          [[NMIN:%.*]] = icmp ne i32 %a, -2147483648
// ON-NEXT:     [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 %a, i32 -1, i1 [[NMIN]])
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       [[NMIN:%.*]] = icmp ne i32 %a, -2147483648
// GUARD-NEXT:  [[D:%.*]] = select i1 [[NMIN]], i32 -1, i32 1
// GUARD-NEXT:  sdiv i32 %a, [[D]]
// OFF:         sdiv i32 %a, -1
int f_div_minus_one(int a) { return a / -1; }

// CHECK-LABEL: @{{.*}}f_div_minus_one_const
// CHECK-NOT:   select
// CHECK:       ret i32 -5
int f_div_minus_one_const(void) { return 5 / -1; }

// Division by a constant zero has an unspecified result too.
// CHECK-LABEL: @{{.*}}f_div_zero
// ON:          [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 %a, i32 0, i1 false)
// ON-NEXT:     freeze i32 [[Q]]
// GUARD:       sdiv i32 %a, 1
// OFF:         sdiv i32 %a, 0
int f_div_zero(int a) { return a / 0; }

// A constant operand that is not a plain integer, such as the address of an
// object, is not known to be safe.
__constant int obj = 1;

// CHECK-LABEL: @{{.*}}f_sdiv_by_address
// ON:          [[Q:%.*]] = call i64 @llvm.masked.sdiv.i64(i64 %a, i64 ptrtoint (ptr{{.*}} @obj to i64), i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i64 ptrtoint (ptr{{.*}} @obj to i64), i64 1
// GUARD-NEXT:  sdiv i64 %a, [[D]]
// OFF:         sdiv i64 %a, ptrtoint (ptr{{.*}} @obj to i64)
long f_sdiv_by_address(long a) { return a / (long)&obj; }

// CHECK-LABEL: @{{.*}}f_srem_of_address
// ON:          [[Q:%.*]] = call i64 @llvm.masked.srem.i64(i64 ptrtoint (ptr{{.*}} @obj to i64), i64 %b, i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       [[D:%.*]] = select i1 {{.*}}, i64 %b, i64 1
// GUARD-NEXT:  srem i64 ptrtoint (ptr{{.*}} @obj to i64), [[D]]
// OFF:         srem i64 ptrtoint (ptr{{.*}} @obj to i64), %b
long f_srem_of_address(long b) { return (long)&obj % b; }

// CHECK-LABEL: @{{.*}}f_udiv_by_address
// ON:          [[Q:%.*]] = call i64 @llvm.masked.udiv.i64(i64 %a, i64 ptrtoint (ptr{{.*}} @obj to i64), i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       udiv i64 %a,
// OFF:         udiv i64 %a, ptrtoint (ptr{{.*}} @obj to i64)
ulong f_udiv_by_address(ulong a) { return a / (ulong)&obj; }

// CHECK-LABEL: @{{.*}}f_urem_by_address
// ON:          [[Q:%.*]] = call i64 @llvm.masked.urem.i64(i64 %a, i64 ptrtoint (ptr{{.*}} @obj to i64), i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
// GUARD:       urem i64 %a,
// OFF:         urem i64 %a, ptrtoint (ptr{{.*}} @obj to i64)
ulong f_urem_by_address(ulong a) { return a % (ulong)&obj; }
