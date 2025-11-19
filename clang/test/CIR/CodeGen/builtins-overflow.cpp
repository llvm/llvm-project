// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=OGCG --input-file=%t.ll

bool test_add_overflow_uint_uint_uint(unsigned x, unsigned y, unsigned *res) {
  return __builtin_add_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z32test_add_overflow_uint_uint_uintjjPj
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#LHS]], %[[#RHS]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_add_overflow_int_int_int(int x, int y, int *res) {
  return __builtin_add_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z29test_add_overflow_int_int_intiiPi
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#LHS]], %[[#RHS]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_add_overflow_xint31_xint31_xint31(_BitInt(31) x, _BitInt(31) y, _BitInt(31) *res) {
  return __builtin_add_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z38test_add_overflow_xint31_xint31_xint31DB31_S_PS_
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.int<s, 31>>>, !cir.ptr<!cir.int<s, 31>>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#LHS]], %[[#RHS]]) : <s, 31>, (<s, 31>, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !cir.int<s, 31>, !cir.ptr<!cir.int<s, 31>>
//      CIR: }

bool test_sub_overflow_uint_uint_uint(unsigned x, unsigned y, unsigned *res) {
  return __builtin_sub_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z32test_sub_overflow_uint_uint_uintjjPj
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#LHS]], %[[#RHS]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_sub_overflow_int_int_int(int x, int y, int *res) {
  return __builtin_sub_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z29test_sub_overflow_int_int_intiiPi
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#LHS]], %[[#RHS]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_sub_overflow_xint31_xint31_xint31(_BitInt(31) x, _BitInt(31) y, _BitInt(31) *res) {
  return __builtin_sub_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z38test_sub_overflow_xint31_xint31_xint31DB31_S_PS_
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.int<s, 31>>>, !cir.ptr<!cir.int<s, 31>>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#LHS]], %[[#RHS]]) : <s, 31>, (<s, 31>, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !cir.int<s, 31>, !cir.ptr<!cir.int<s, 31>>
//      CIR: }

bool test_mul_overflow_uint_uint_uint(unsigned x, unsigned y, unsigned *res) {
  return __builtin_mul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z32test_mul_overflow_uint_uint_uintjjPj
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#LHS]], %[[#RHS]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_mul_overflow_int_int_int(int x, int y, int *res) {
  return __builtin_mul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z29test_mul_overflow_int_int_intiiPi
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#LHS]], %[[#RHS]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_mul_overflow_xint31_xint31_xint31(_BitInt(31) x, _BitInt(31) y, _BitInt(31) *res) {
  return __builtin_mul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z38test_mul_overflow_xint31_xint31_xint31DB31_S_PS_
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.int<s, 31>>, !cir.int<s, 31>
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.int<s, 31>>>, !cir.ptr<!cir.int<s, 31>>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#LHS]], %[[#RHS]]) : <s, 31>, (<s, 31>, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !cir.int<s, 31>, !cir.ptr<!cir.int<s, 31>>
//      CIR: }

bool test_mul_overflow_ulong_ulong_long(unsigned long x, unsigned long y, unsigned long *res) {
  return __builtin_mul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z34test_mul_overflow_ulong_ulong_longmmPm
//      CIR:   %[[#LHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RHS:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#LHS]], %[[#RHS]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_add_overflow_uint_int_int(unsigned x, int y, int *res) {
  return __builtin_add_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z30test_add_overflow_uint_int_intjiPi
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[#PROM_X:]] = cir.cast integral %[[#X]] : !u32i -> !cir.int<s, 33>
// CIR-NEXT:   %[[#PROM_Y:]] = cir.cast integral %[[#Y]] : !s32i -> !cir.int<s, 33>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#PROM_X]], %[[#PROM_Y]]) : <s, 33>, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_add_overflow_volatile(int x, int y, volatile int *res) {
  return __builtin_add_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z26test_add_overflow_volatileiiPVi
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store volatile{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_uadd_overflow(unsigned x, unsigned y, unsigned *res) {
  return __builtin_uadd_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_uadd_overflowjjPj
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_uaddl_overflow(unsigned long x, unsigned long y, unsigned long *res) {
  return __builtin_uaddl_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_uaddl_overflowmmPm
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_uaddll_overflow(unsigned long long x, unsigned long long y, unsigned long long *res) {
  return __builtin_uaddll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_uaddll_overflowyyPy
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_usub_overflow(unsigned x, unsigned y, unsigned *res) {
  return __builtin_usub_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_usub_overflowjjPj
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_usubl_overflow(unsigned long x, unsigned long y, unsigned long *res) {
  return __builtin_usubl_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_usubl_overflowmmPm
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_usubll_overflow(unsigned long long x, unsigned long long y, unsigned long long *res) {
  return __builtin_usubll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_usubll_overflowyyPy
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_umul_overflow(unsigned x, unsigned y, unsigned *res) {
  return __builtin_umul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_umul_overflowjjPj
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !u32i, (!u32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u32i, !cir.ptr<!u32i>
//      CIR: }

bool test_umull_overflow(unsigned long x, unsigned long y, unsigned long *res) {
  return __builtin_umull_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_umull_overflowmmPm
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_umulll_overflow(unsigned long long x, unsigned long long y, unsigned long long *res) {
  return __builtin_umulll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_umulll_overflowyyPy
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !u64i, (!u64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !u64i, !cir.ptr<!u64i>
//      CIR: }

bool test_sadd_overflow(int x, int y, int *res) {
  return __builtin_sadd_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_sadd_overflowiiPi
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_saddl_overflow(long x, long y, long *res) {
  return __builtin_saddl_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_saddl_overflowllPl
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }

bool test_saddll_overflow(long long x, long long y, long long *res) {
  return __builtin_saddll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_saddll_overflowxxPx
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(add, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }

bool test_ssub_overflow(int x, int y, int *res) {
  return __builtin_ssub_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_ssub_overflowiiPi
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_ssubl_overflow(long x, long y, long *res) {
  return __builtin_ssubl_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_ssubl_overflowllPl
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }

bool test_ssubll_overflow(long long x, long long y, long long *res) {
  return __builtin_ssubll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_ssubll_overflowxxPx
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(sub, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }

bool test_smul_overflow(int x, int y, int *res) {
  return __builtin_smul_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z18test_smul_overflowiiPi
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !s32i, (!s32i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s32i, !cir.ptr<!s32i>
//      CIR: }

bool test_smull_overflow(long x, long y, long *res) {
  return __builtin_smull_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z19test_smull_overflowllPl
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }

bool test_smulll_overflow(long long x, long long y, long long *res) {
  return __builtin_smulll_overflow(x, y, res);
}

//      CIR: cir.func dso_local @_Z20test_smulll_overflowxxPx
//      CIR:   %[[#X:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#Y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s64i>, !s64i
// CIR-NEXT:   %[[#RES_PTR:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CIR-NEXT:   %[[RES:.+]], %{{.+}} = cir.binop.overflow(mul, %[[#X]], %[[#Y]]) : !s64i, (!s64i, !cir.bool)
// CIR-NEXT:   cir.store{{.*}} %[[RES]], %[[#RES_PTR]] : !s64i, !cir.ptr<!s64i>
//      CIR: }
