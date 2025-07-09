! Test lowering complex division to llvm ir according to options

! REQUIRES: target=x86_64{{.*}}
! RUN: %flang -fcomplex-arithmetic=improved -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: %flang -fcomplex-arithmetic=basic -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC


! CHECK-LABEL: @div_test_extended
! CHECK-SAME: ptr %[[RET:.*]], ptr %[[LHS:.*]], ptr %[[RHS:.*]])
! CHECK: %[[LOAD_LHS:.*]] = load { x86_fp80, x86_fp80 }, ptr %[[LHS]], align 16
! CHECK: %[[LOAD_RHS:.*]] = load { x86_fp80, x86_fp80 }, ptr %[[RHS]], align 16
! CHECK: %[[LHS_REAL:.*]] = extractvalue { x86_fp80, x86_fp80 } %[[LOAD_LHS]], 0
! CHECK: %[[LHS_IMAG:.*]] = extractvalue { x86_fp80, x86_fp80 } %[[LOAD_LHS]], 1
! CHECK: %[[RHS_REAL:.*]] = extractvalue { x86_fp80, x86_fp80 } %[[LOAD_RHS]], 0
! CHECK: %[[RHS_IMAG:.*]] = extractvalue { x86_fp80, x86_fp80 } %[[LOAD_RHS]], 1

! IMPRVD: %[[RHS_REAL_IMAG_RATIO:.*]] = fdiv contract x86_fp80 %[[RHS_REAL]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract x86_fp80 %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_REAL_IMAG_DENOM:.*]] = fadd contract x86_fp80 %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_1:.*]] = fadd contract x86_fp80 %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_1:.*]] = fdiv contract x86_fp80 %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_1:.*]] = fsub contract x86_fp80 %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_1:.*]] = fdiv contract x86_fp80 %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[RHS_IMAG_REAL_RATIO:.*]] = fdiv contract x86_fp80 %[[RHS_IMAG]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract x86_fp80 %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_IMAG_REAL_DENOM:.*]] = fadd contract x86_fp80 %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_2:.*]] = fadd contract x86_fp80 %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_REAL_2:.*]] = fdiv contract x86_fp80 %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_2:.*]] = fsub contract x86_fp80 %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_IMAG_2:.*]] = fdiv contract x86_fp80 %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]

! Case 1. Zero denominator, numerator contains at most one NaN value.
! IMPRVD: %[[RHS_REAL_ABS:.*]] = call contract x86_fp80 @llvm.fabs.f80(x86_fp80 %[[RHS_REAL]])
! IMPRVD: %[[RHS_REAL_ABS_IS_ZERO:.*]] = fcmp oeq x86_fp80 %[[RHS_REAL_ABS]], 0xK00000000000000000000
! IMPRVD: %[[RHS_IMAG_ABS:.*]] = call contract x86_fp80 @llvm.fabs.f80(x86_fp80 %[[RHS_IMAG]])
! IMPRVD: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = fcmp oeq x86_fp80 %[[RHS_IMAG_ABS]], 0xK00000000000000000000
! IMPRVD: %[[LHS_REAL_IS_NOT_NAN:.*]] = fcmp ord x86_fp80 %[[LHS_REAL]], 0xK00000000000000000000
! IMPRVD: %[[LHS_IMAG_IS_NOT_NAN:.*]] = fcmp ord x86_fp80 %[[LHS_IMAG]], 0xK00000000000000000000
! IMPRVD: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or i1 %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]]
! IMPRVD: %[[RHS_IS_ZERO:.*]] = and i1 %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]]
! IMPRVD: %[[RESULT_IS_INFINITY:.*]] = and i1 %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]]
! IMPRVD: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = call x86_fp80 @llvm.copysign.f80(x86_fp80 0xK7FFF8000000000000000, x86_fp80 %[[RHS_REAL]])
! IMPRVD: %[[INFINITY_RESULT_REAL:.*]] = fmul contract x86_fp80 %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]]
! IMPRVD: %[[INFINITY_RESULT_IMAG:.*]] = fmul contract x86_fp80 %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]]

! Case 2. Infinite numerator, finite denominator.
! IMPRVD: %[[RHS_REAL_FINITE:.*]] = fcmp one x86_fp80 %[[RHS_REAL_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[RHS_IMAG_FINITE:.*]] = fcmp one x86_fp80 %[[RHS_IMAG_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[RHS_IS_FINITE:.*]] = and i1 %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]]
! IMPRVD: %[[LHS_REAL_ABS:.*]] = call contract x86_fp80 @llvm.fabs.f80(x86_fp80 %[[LHS_REAL]])
! IMPRVD: %[[LHS_REAL_INFINITE:.*]] = fcmp oeq x86_fp80 %[[LHS_REAL_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[LHS_IMAG_ABS:.*]] = call contract x86_fp80 @llvm.fabs.f80(x86_fp80 %[[LHS_IMAG]])
! IMPRVD: %[[LHS_IMAG_INFINITE:.*]] = fcmp oeq x86_fp80 %[[LHS_IMAG_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[LHS_IS_INFINITE:.*]] = or i1 %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]]
! IMPRVD: %[[INF_NUM_FINITE_DENOM:.*]] = and i1 %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]]
! IMPRVD: %[[LHS_REAL_IS_INF:.*]] = select i1 %[[LHS_REAL_INFINITE]], x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK00000000000000000000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = call x86_fp80 @llvm.copysign.f80(x86_fp80 %[[LHS_REAL_IS_INF]], x86_fp80 %[[LHS_REAL]])
! IMPRVD: %[[LHS_IMAG_IS_INF:.*]] = select i1 %[[LHS_IMAG_INFINITE]], x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK00000000000000000000
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = call x86_fp80 @llvm.copysign.f80(x86_fp80 %[[LHS_IMAG_IS_INF]], x86_fp80 %[[LHS_IMAG]])
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract x86_fp80 %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract x86_fp80 %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[INF_MULTIPLICATOR_1:.*]] = fadd contract x86_fp80 %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_3:.*]] = fmul contract x86_fp80 %[[INF_MULTIPLICATOR_1]], 0xK7FFF8000000000000000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract x86_fp80 %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract x86_fp80 %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[INF_MULTIPLICATOR_2:.*]] = fsub contract x86_fp80 %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_IMAG_3:.*]] = fmul contract x86_fp80 %[[INF_MULTIPLICATOR_2]], 0xK7FFF8000000000000000

! Case 3. Finite numerator, infinite denominator.
! IMPRVD: %[[LHS_REAL_FINITE:.*]] = fcmp one x86_fp80 %[[LHS_REAL_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[LHS_IMAG_FINITE:.*]] = fcmp one x86_fp80 %[[LHS_IMAG_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[LHS_IS_FINITE:.*]] = and i1 %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]]
! IMPRVD: %[[RHS_REAL_INFINITE:.*]] = fcmp oeq x86_fp80 %[[RHS_REAL_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[RHS_IMAG_INFINITE:.*]] = fcmp oeq x86_fp80 %[[RHS_IMAG_ABS]], 0xK7FFF8000000000000000
! IMPRVD: %[[RHS_IS_INFINITE:.*]] = or i1 %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]]
! IMPRVD: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and i1 %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]]
! IMPRVD: %[[RHS_REAL_IS_INF:.*]] = select i1 %[[RHS_REAL_INFINITE]], x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK00000000000000000000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = call x86_fp80 @llvm.copysign.f80(x86_fp80 %[[RHS_REAL_IS_INF]], x86_fp80 %[[RHS_REAL]])
! IMPRVD: %[[RHS_IMAG_IS_INF:.*]] = select i1 %[[RHS_IMAG_INFINITE]], x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK00000000000000000000
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = call x86_fp80 @llvm.copysign.f80(x86_fp80 %[[RHS_IMAG_IS_INF]], x86_fp80 %[[RHS_IMAG]])
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_1:.*]] = fadd contract x86_fp80 %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_4:.*]] = fmul contract x86_fp80 %[[ZERO_MULTIPLICATOR_1]], 0xK00000000000000000000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_2:.*]] = fsub contract x86_fp80 %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_4:.*]] = fmul contract x86_fp80 %[[ZERO_MULTIPLICATOR_2]], 0xK00000000000000000000

! IMPRVD: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = fcmp olt x86_fp80 %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]]
! IMPRVD: %[[RESULT_REAL:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], x86_fp80 %[[RESULT_REAL_1]], x86_fp80 %[[RESULT_REAL_2]]
! IMPRVD: %[[RESULT_IMAG:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], x86_fp80 %[[RESULT_IMAG_1]], x86_fp80 %[[RESULT_IMAG_2]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], x86_fp80 %[[RESULT_REAL_4]], x86_fp80 %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], x86_fp80 %[[RESULT_IMAG_4]], x86_fp80 %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], x86_fp80 %[[RESULT_REAL_3]], x86_fp80 %[[RESULT_REAL_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], x86_fp80 %[[RESULT_IMAG_3]], x86_fp80 %[[RESULT_IMAG_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], x86_fp80 %[[INFINITY_RESULT_REAL]], x86_fp80 %[[RESULT_REAL_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], x86_fp80 %[[INFINITY_RESULT_IMAG]], x86_fp80 %[[RESULT_IMAG_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_REAL_IS_NAN:.*]] = fcmp uno x86_fp80 %[[RESULT_REAL]], 0xK00000000000000000000
! IMPRVD: %[[RESULT_IMAG_IS_NAN:.*]] = fcmp uno x86_fp80 %[[RESULT_IMAG]], 0xK00000000000000000000
! IMPRVD: %[[RESULT_IS_NAN:.*]] = and i1 %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]]
! IMPRVD: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], x86_fp80 %[[RESULT_REAL_SPECIAL_CASE_1]], x86_fp80 %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], x86_fp80 %[[RESULT_IMAG_SPECIAL_CASE_1]], x86_fp80 %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_1:.*]] = insertvalue { x86_fp80, x86_fp80 } poison, x86_fp80 %[[RESULT_REAL_WITH_SPECIAL_CASES]], 0
! IMPRVD: %[[RESULT_2:.*]] = insertvalue { x86_fp80, x86_fp80 } %[[RESULT_1]], x86_fp80 %[[RESULT_IMAG_WITH_SPECIAL_CASES]], 1
! IMPRVD: store { x86_fp80, x86_fp80 } %[[RESULT_2]], ptr %[[RET]], align 16

! BASIC-DAG: %[[RHS_REAL_SQ:.*]] = fmul contract x86_fp80 %[[RHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[RHS_IMAG_SQ:.*]] = fmul contract x86_fp80 %[[RHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[SQ_NORM:.*]] = fadd contract x86_fp80 %[[RHS_REAL_SQ]], %[[RHS_IMAG_SQ]]
! BASIC-DAG: %[[REAL_TMP_0:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[REAL_TMP_1:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[REAL_TMP_2:.*]] = fadd contract x86_fp80 %[[REAL_TMP_0]], %[[REAL_TMP_1]]
! BASIC-DAG: %[[IMAG_TMP_0:.*]] = fmul contract x86_fp80 %[[LHS_IMAG]], %[[RHS_REAL]]
! BASIC-DAG: %[[IMAG_TMP_1:.*]] = fmul contract x86_fp80 %[[LHS_REAL]], %[[RHS_IMAG]]
! BASIC: %[[IMAG_TMP_2:.*]] = fsub contract x86_fp80 %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]
! BASIC: %[[REAL:.*]] = fdiv contract x86_fp80 %[[REAL_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[IMAG:.*]] = fdiv contract x86_fp80 %[[IMAG_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[RESULT_1:.*]] = insertvalue { x86_fp80, x86_fp80 } poison, x86_fp80 %[[REAL]], 0
! BASIC: %[[RESULT_2:.*]] = insertvalue { x86_fp80, x86_fp80 } %[[RESULT_1]], x86_fp80 %[[IMAG]], 1
! BASIC: store { x86_fp80, x86_fp80 } %[[RESULT_2]], ptr %[[RET]], align 16

! CHECK: ret void
subroutine div_test_extended(a,b,c)
  complex(kind=10) :: a, b, c
  a = b / c
end subroutine div_test_extended
