! Test lowering complex division to llvm ir according to options

! RUN: %flang -fcomplex-arithmetic=improved -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: %flang -fcomplex-arithmetic=basic -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC


! CHECK-LABEL: @div_test_half
! CHECK-SAME: ptr noalias %[[RET:.*]], ptr noalias %[[LHS:.*]], ptr noalias %[[RHS:.*]])
! CHECK: %[[LOAD_LHS:.*]] = load { half, half }, ptr %[[LHS]], align 2
! CHECK: %[[LOAD_RHS:.*]] = load { half, half }, ptr %[[RHS]], align 2
! CHECK: %[[LHS_REAL:.*]] = extractvalue { half, half } %[[LOAD_LHS]], 0
! CHECK: %[[LHS_IMAG:.*]] = extractvalue { half, half } %[[LOAD_LHS]], 1
! CHECK: %[[RHS_REAL:.*]] = extractvalue { half, half } %[[LOAD_RHS]], 0
! CHECK: %[[RHS_IMAG:.*]] = extractvalue { half, half } %[[LOAD_RHS]], 1

! IMPRVD: %[[RHS_REAL_IMAG_RATIO:.*]] = fdiv contract half %[[RHS_REAL]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract half %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_REAL_IMAG_DENOM:.*]] = fadd contract half %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_1:.*]] = fadd contract half %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_1:.*]] = fdiv contract half %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_1:.*]] = fsub contract half %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_1:.*]] = fdiv contract half %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[RHS_IMAG_REAL_RATIO:.*]] = fdiv contract half %[[RHS_IMAG]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract half %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_IMAG_REAL_DENOM:.*]] = fadd contract half %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_2:.*]] = fadd contract half %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_REAL_2:.*]] = fdiv contract half %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_2:.*]] = fsub contract half %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_IMAG_2:.*]] = fdiv contract half %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]

! Case 1. Zero denominator, numerator contains at most one NaN value.
! IMPRVD: %[[RHS_REAL_ABS:.*]] = call contract half @llvm.fabs.f16(half %[[RHS_REAL]])
! IMPRVD: %[[RHS_REAL_ABS_IS_ZERO:.*]] = fcmp oeq half %[[RHS_REAL_ABS]], 0xH0000
! IMPRVD: %[[RHS_IMAG_ABS:.*]] = call contract half @llvm.fabs.f16(half %[[RHS_IMAG]])
! IMPRVD: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = fcmp oeq half %[[RHS_IMAG_ABS]], 0xH0000
! IMPRVD: %[[LHS_REAL_IS_NOT_NAN:.*]] = fcmp ord half %[[LHS_REAL]], 0xH0000
! IMPRVD: %[[LHS_IMAG_IS_NOT_NAN:.*]] = fcmp ord half %[[LHS_IMAG]], 0xH0000
! IMPRVD: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or i1 %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]]
! IMPRVD: %[[RHS_IS_ZERO:.*]] = and i1 %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]]
! IMPRVD: %[[RESULT_IS_INFINITY:.*]] = and i1 %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]]
! IMPRVD: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = call half @llvm.copysign.f16(half 0xH7C00, half %[[RHS_REAL]])
! IMPRVD: %[[INFINITY_RESULT_REAL:.*]] = fmul contract half %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]]
! IMPRVD: %[[INFINITY_RESULT_IMAG:.*]] = fmul contract half %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]]

! Case 2. Infinite numerator, finite denominator.
! IMPRVD: %[[RHS_REAL_FINITE:.*]] = fcmp one half %[[RHS_REAL_ABS]], 0xH7C00
! IMPRVD: %[[RHS_IMAG_FINITE:.*]] = fcmp one half %[[RHS_IMAG_ABS]], 0xH7C00
! IMPRVD: %[[RHS_IS_FINITE:.*]] = and i1 %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]]
! IMPRVD: %[[LHS_REAL_ABS:.*]] = call contract half @llvm.fabs.f16(half %[[LHS_REAL]])
! IMPRVD: %[[LHS_REAL_INFINITE:.*]] = fcmp oeq half %[[LHS_REAL_ABS]], 0xH7C00
! IMPRVD: %[[LHS_IMAG_ABS:.*]] = call contract half @llvm.fabs.f16(half %[[LHS_IMAG]])
! IMPRVD: %[[LHS_IMAG_INFINITE:.*]] = fcmp oeq half %[[LHS_IMAG_ABS]], 0xH7C00
! IMPRVD: %[[LHS_IS_INFINITE:.*]] = or i1 %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]]
! IMPRVD: %[[INF_NUM_FINITE_DENOM:.*]] = and i1 %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]]
! IMPRVD: %[[LHS_REAL_IS_INF:.*]] = select i1 %[[LHS_REAL_INFINITE]], half 0xH3C00, half 0xH0000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = call half @llvm.copysign.f16(half %[[LHS_REAL_IS_INF]], half %[[LHS_REAL]])
! IMPRVD: %[[LHS_IMAG_IS_INF:.*]] = select i1 %[[LHS_IMAG_INFINITE]], half 0xH3C00, half 0xH0000
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = call half @llvm.copysign.f16(half %[[LHS_IMAG_IS_INF]], half %[[LHS_IMAG]])
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract half %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract half %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[INF_MULTIPLICATOR_1:.*]] = fadd contract half %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_3:.*]] = fmul contract half %[[INF_MULTIPLICATOR_1]], 0xH7C00
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract half %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract half %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[INF_MULTIPLICATOR_2:.*]] = fsub contract half %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_IMAG_3:.*]] = fmul contract half %[[INF_MULTIPLICATOR_2]], 0xH7C00

! Case 3. Finite numerator, infinite denominator.
! IMPRVD: %[[LHS_REAL_FINITE:.*]] = fcmp one half %[[LHS_REAL_ABS]], 0xH7C00
! IMPRVD: %[[LHS_IMAG_FINITE:.*]] = fcmp one half %[[LHS_IMAG_ABS]], 0xH7C00
! IMPRVD: %[[LHS_IS_FINITE:.*]] = and i1 %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]]
! IMPRVD: %[[RHS_REAL_INFINITE:.*]] = fcmp oeq half %[[RHS_REAL_ABS]], 0xH7C00
! IMPRVD: %[[RHS_IMAG_INFINITE:.*]] = fcmp oeq half %[[RHS_IMAG_ABS]], 0xH7C00
! IMPRVD: %[[RHS_IS_INFINITE:.*]] = or i1 %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]]
! IMPRVD: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and i1 %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]]
! IMPRVD: %[[RHS_REAL_IS_INF:.*]] = select i1 %[[RHS_REAL_INFINITE]], half 0xH3C00, half 0xH0000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = call half @llvm.copysign.f16(half %[[RHS_REAL_IS_INF]], half %[[RHS_REAL]])
! IMPRVD: %[[RHS_IMAG_IS_INF:.*]] = select i1 %[[RHS_IMAG_INFINITE]], half 0xH3C00, half 0xH0000
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = call half @llvm.copysign.f16(half %[[RHS_IMAG_IS_INF]], half %[[RHS_IMAG]])
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_1:.*]] = fadd contract half %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_4:.*]] = fmul contract half %[[ZERO_MULTIPLICATOR_1]], 0xH0000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_2:.*]] = fsub contract half %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_4:.*]] = fmul contract half %[[ZERO_MULTIPLICATOR_2]], 0xH0000

! IMPRVD: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = fcmp olt half %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]]
! IMPRVD: %[[RESULT_REAL:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], half %[[RESULT_REAL_1]], half %[[RESULT_REAL_2]]
! IMPRVD: %[[RESULT_IMAG:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], half %[[RESULT_IMAG_1]], half %[[RESULT_IMAG_2]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], half %[[RESULT_REAL_4]], half %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], half %[[RESULT_IMAG_4]], half %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], half %[[RESULT_REAL_3]], half %[[RESULT_REAL_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], half %[[RESULT_IMAG_3]], half %[[RESULT_IMAG_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], half %[[INFINITY_RESULT_REAL]], half %[[RESULT_REAL_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], half %[[INFINITY_RESULT_IMAG]], half %[[RESULT_IMAG_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_REAL_IS_NAN:.*]] = fcmp uno half %[[RESULT_REAL]], 0xH0000
! IMPRVD: %[[RESULT_IMAG_IS_NAN:.*]] = fcmp uno half %[[RESULT_IMAG]], 0xH0000
! IMPRVD: %[[RESULT_IS_NAN:.*]] = and i1 %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]]
! IMPRVD: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], half %[[RESULT_REAL_SPECIAL_CASE_1]], half %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], half %[[RESULT_IMAG_SPECIAL_CASE_1]], half %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_1:.*]] = insertvalue { half, half } poison, half %[[RESULT_REAL_WITH_SPECIAL_CASES]], 0
! IMPRVD: %[[RESULT_2:.*]] = insertvalue { half, half } %[[RESULT_1]], half %[[RESULT_IMAG_WITH_SPECIAL_CASES]], 1
! IMPRVD: store { half, half } %[[RESULT_2]], ptr %[[RET]], align 2

! BASIC-DAG: %[[RHS_REAL_SQ:.*]] = fmul contract half %[[RHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[RHS_IMAG_SQ:.*]] = fmul contract half %[[RHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[SQ_NORM:.*]] = fadd contract half %[[RHS_REAL_SQ]], %[[RHS_IMAG_SQ]]
! BASIC-DAG: %[[REAL_TMP_0:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[REAL_TMP_1:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[REAL_TMP_2:.*]] = fadd contract half %[[REAL_TMP_0]], %[[REAL_TMP_1]]
! BASIC-DAG: %[[IMAG_TMP_0:.*]] = fmul contract half %[[LHS_IMAG]], %[[RHS_REAL]]
! BASIC-DAG: %[[IMAG_TMP_1:.*]] = fmul contract half %[[LHS_REAL]], %[[RHS_IMAG]]
! BASIC: %[[IMAG_TMP_2:.*]] = fsub contract half %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]
! BASIC: %[[REAL:.*]] = fdiv contract half %[[REAL_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[IMAG:.*]] = fdiv contract half %[[IMAG_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[RESULT_1:.*]] = insertvalue { half, half } poison, half %[[REAL]], 0
! BASIC: %[[RESULT_2:.*]] = insertvalue { half, half } %[[RESULT_1]], half %[[IMAG]], 1
! BASIC: store { half, half } %[[RESULT_2]], ptr %[[RET]], align 2

! CHECK: ret void
subroutine div_test_half(a,b,c)
  complex(kind=2) :: a, b, c
  a = b / c
end subroutine div_test_half


! CHECK-LABEL: @div_test_bfloat
! CHECK-SAME: ptr noalias %[[RET:.*]], ptr noalias %[[LHS:.*]], ptr noalias %[[RHS:.*]])
! CHECK: %[[LOAD_LHS:.*]] = load { bfloat, bfloat }, ptr %[[LHS]], align 2
! CHECK: %[[LOAD_RHS:.*]] = load { bfloat, bfloat }, ptr %[[RHS]], align 2
! CHECK: %[[LHS_REAL:.*]] = extractvalue { bfloat, bfloat } %[[LOAD_LHS]], 0
! CHECK: %[[LHS_IMAG:.*]] = extractvalue { bfloat, bfloat } %[[LOAD_LHS]], 1
! CHECK: %[[RHS_REAL:.*]] = extractvalue { bfloat, bfloat } %[[LOAD_RHS]], 0
! CHECK: %[[RHS_IMAG:.*]] = extractvalue { bfloat, bfloat } %[[LOAD_RHS]], 1

! IMPRVD: %[[RHS_REAL_IMAG_RATIO:.*]] = fdiv contract bfloat %[[RHS_REAL]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract bfloat %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_REAL_IMAG_DENOM:.*]] = fadd contract bfloat %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_1:.*]] = fadd contract bfloat %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_1:.*]] = fdiv contract bfloat %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_1:.*]] = fsub contract bfloat %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_1:.*]] = fdiv contract bfloat %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[RHS_IMAG_REAL_RATIO:.*]] = fdiv contract bfloat %[[RHS_IMAG]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract bfloat %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_IMAG_REAL_DENOM:.*]] = fadd contract bfloat %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_2:.*]] = fadd contract bfloat %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_REAL_2:.*]] = fdiv contract bfloat %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_2:.*]] = fsub contract bfloat %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_IMAG_2:.*]] = fdiv contract bfloat %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]

! Case 1. Zero denominator, numerator contains at most one NaN value.
! IMPRVD: %[[RHS_REAL_ABS:.*]] = call contract bfloat @llvm.fabs.bf16(bfloat %[[RHS_REAL]])
! IMPRVD: %[[RHS_REAL_ABS_IS_ZERO:.*]] = fcmp oeq bfloat %[[RHS_REAL_ABS]], 0xR0000
! IMPRVD: %[[RHS_IMAG_ABS:.*]] = call contract bfloat @llvm.fabs.bf16(bfloat %[[RHS_IMAG]])
! IMPRVD: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = fcmp oeq bfloat %[[RHS_IMAG_ABS]], 0xR0000
! IMPRVD: %[[LHS_REAL_IS_NOT_NAN:.*]] = fcmp ord bfloat %[[LHS_REAL]], 0xR0000
! IMPRVD: %[[LHS_IMAG_IS_NOT_NAN:.*]] = fcmp ord bfloat %[[LHS_IMAG]], 0xR0000
! IMPRVD: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or i1 %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]]
! IMPRVD: %[[RHS_IS_ZERO:.*]] = and i1 %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]]
! IMPRVD: %[[RESULT_IS_INFINITY:.*]] = and i1 %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]]
! IMPRVD: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = call bfloat @llvm.copysign.bf16(bfloat 0xR7F80, bfloat %[[RHS_REAL]])
! IMPRVD: %[[INFINITY_RESULT_REAL:.*]] = fmul contract bfloat %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]]
! IMPRVD: %[[INFINITY_RESULT_IMAG:.*]] = fmul contract bfloat %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]]

! Case 2. Infinite numerator, finite denominator.
! IMPRVD: %[[RHS_REAL_FINITE:.*]] = fcmp one bfloat %[[RHS_REAL_ABS]], 0xR7F80
! IMPRVD: %[[RHS_IMAG_FINITE:.*]] = fcmp one bfloat %[[RHS_IMAG_ABS]], 0xR7F80
! IMPRVD: %[[RHS_IS_FINITE:.*]] = and i1 %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]]
! IMPRVD: %[[LHS_REAL_ABS:.*]] = call contract bfloat @llvm.fabs.bf16(bfloat %[[LHS_REAL]])
! IMPRVD: %[[LHS_REAL_INFINITE:.*]] = fcmp oeq bfloat %[[LHS_REAL_ABS]], 0xR7F80
! IMPRVD: %[[LHS_IMAG_ABS:.*]] = call contract bfloat @llvm.fabs.bf16(bfloat %[[LHS_IMAG]])
! IMPRVD: %[[LHS_IMAG_INFINITE:.*]] = fcmp oeq bfloat %[[LHS_IMAG_ABS]], 0xR7F80
! IMPRVD: %[[LHS_IS_INFINITE:.*]] = or i1 %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]]
! IMPRVD: %[[INF_NUM_FINITE_DENOM:.*]] = and i1 %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]]
! IMPRVD: %[[LHS_REAL_IS_INF:.*]] = select i1 %[[LHS_REAL_INFINITE]], bfloat 0xR3F80, bfloat 0xR0000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = call bfloat @llvm.copysign.bf16(bfloat %[[LHS_REAL_IS_INF]], bfloat %[[LHS_REAL]])
! IMPRVD: %[[LHS_IMAG_IS_INF:.*]] = select i1 %[[LHS_IMAG_INFINITE]], bfloat 0xR3F80, bfloat 0xR0000
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = call bfloat @llvm.copysign.bf16(bfloat %[[LHS_IMAG_IS_INF]], bfloat %[[LHS_IMAG]])
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract bfloat %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract bfloat %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[INF_MULTIPLICATOR_1:.*]] = fadd contract bfloat %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_3:.*]] = fmul contract bfloat %[[INF_MULTIPLICATOR_1]], 0xR7F80
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract bfloat %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract bfloat %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[INF_MULTIPLICATOR_2:.*]] = fsub contract bfloat %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_IMAG_3:.*]] = fmul contract bfloat %[[INF_MULTIPLICATOR_2]], 0xR7F80

! Case 3. Finite numerator, infinite denominator.
! IMPRVD: %[[LHS_REAL_FINITE:.*]] = fcmp one bfloat %[[LHS_REAL_ABS]], 0xR7F80
! IMPRVD: %[[LHS_IMAG_FINITE:.*]] = fcmp one bfloat %[[LHS_IMAG_ABS]], 0xR7F80
! IMPRVD: %[[LHS_IS_FINITE:.*]] = and i1 %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]]
! IMPRVD: %[[RHS_REAL_INFINITE:.*]] = fcmp oeq bfloat %[[RHS_REAL_ABS]], 0xR7F80
! IMPRVD: %[[RHS_IMAG_INFINITE:.*]] = fcmp oeq bfloat %[[RHS_IMAG_ABS]], 0xR7F80
! IMPRVD: %[[RHS_IS_INFINITE:.*]] = or i1 %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]]
! IMPRVD: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and i1 %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]]
! IMPRVD: %[[RHS_REAL_IS_INF:.*]] = select i1 %[[RHS_REAL_INFINITE]], bfloat 0xR3F80, bfloat 0xR0000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = call bfloat @llvm.copysign.bf16(bfloat %[[RHS_REAL_IS_INF]], bfloat %[[RHS_REAL]])
! IMPRVD: %[[RHS_IMAG_IS_INF:.*]] = select i1 %[[RHS_IMAG_INFINITE]], bfloat 0xR3F80, bfloat 0xR0000
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = call bfloat @llvm.copysign.bf16(bfloat %[[RHS_IMAG_IS_INF]], bfloat %[[RHS_IMAG]])
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_1:.*]] = fadd contract bfloat %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_4:.*]] = fmul contract bfloat %[[ZERO_MULTIPLICATOR_1]], 0xR0000
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_2:.*]] = fsub contract bfloat %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_4:.*]] = fmul contract bfloat %[[ZERO_MULTIPLICATOR_2]], 0xR0000

! IMPRVD: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = fcmp olt bfloat %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]]
! IMPRVD: %[[RESULT_REAL:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], bfloat %[[RESULT_REAL_1]], bfloat %[[RESULT_REAL_2]]
! IMPRVD: %[[RESULT_IMAG:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], bfloat %[[RESULT_IMAG_1]], bfloat %[[RESULT_IMAG_2]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], bfloat %[[RESULT_REAL_4]], bfloat %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], bfloat %[[RESULT_IMAG_4]], bfloat %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], bfloat %[[RESULT_REAL_3]], bfloat %[[RESULT_REAL_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], bfloat %[[RESULT_IMAG_3]], bfloat %[[RESULT_IMAG_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], bfloat %[[INFINITY_RESULT_REAL]], bfloat %[[RESULT_REAL_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], bfloat %[[INFINITY_RESULT_IMAG]], bfloat %[[RESULT_IMAG_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_REAL_IS_NAN:.*]] = fcmp uno bfloat %[[RESULT_REAL]], 0xR0000
! IMPRVD: %[[RESULT_IMAG_IS_NAN:.*]] = fcmp uno bfloat %[[RESULT_IMAG]], 0xR0000
! IMPRVD: %[[RESULT_IS_NAN:.*]] = and i1 %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]]
! IMPRVD: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], bfloat %[[RESULT_REAL_SPECIAL_CASE_1]], bfloat %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], bfloat %[[RESULT_IMAG_SPECIAL_CASE_1]], bfloat %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_1:.*]] = insertvalue { bfloat, bfloat } poison, bfloat %[[RESULT_REAL_WITH_SPECIAL_CASES]], 0
! IMPRVD: %[[RESULT_2:.*]] = insertvalue { bfloat, bfloat } %[[RESULT_1]], bfloat %[[RESULT_IMAG_WITH_SPECIAL_CASES]], 1
! IMPRVD: store { bfloat, bfloat } %[[RESULT_2]], ptr %[[RET]], align 2

! BASIC-DAG: %[[RHS_REAL_SQ:.*]] = fmul contract bfloat %[[RHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[RHS_IMAG_SQ:.*]] = fmul contract bfloat %[[RHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[SQ_NORM:.*]] = fadd contract bfloat %[[RHS_REAL_SQ]], %[[RHS_IMAG_SQ]]
! BASIC-DAG: %[[REAL_TMP_0:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[REAL_TMP_1:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[REAL_TMP_2:.*]] = fadd contract bfloat %[[REAL_TMP_0]], %[[REAL_TMP_1]]
! BASIC-DAG: %[[IMAG_TMP_0:.*]] = fmul contract bfloat %[[LHS_IMAG]], %[[RHS_REAL]]
! BASIC-DAG: %[[IMAG_TMP_1:.*]] = fmul contract bfloat %[[LHS_REAL]], %[[RHS_IMAG]]
! BASIC: %[[IMAG_TMP_2:.*]] = fsub contract bfloat %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]
! BASIC: %[[REAL:.*]] = fdiv contract bfloat %[[REAL_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[IMAG:.*]] = fdiv contract bfloat %[[IMAG_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[RESULT_1:.*]] = insertvalue { bfloat, bfloat } poison, bfloat %[[REAL]], 0
! BASIC: %[[RESULT_2:.*]] = insertvalue { bfloat, bfloat } %[[RESULT_1]], bfloat %[[IMAG]], 1
! BASIC: store { bfloat, bfloat } %[[RESULT_2]], ptr %[[RET]], align 2

! CHECK: ret void
subroutine div_test_bfloat(a,b,c)
  complex(kind=3) :: a, b, c
  a = b / c
end subroutine div_test_bfloat


! CHECK-LABEL: @div_test_single
! CHECK-SAME: ptr noalias %[[RET:.*]], ptr noalias %[[LHS:.*]], ptr noalias %[[RHS:.*]])
! CHECK: %[[LOAD_LHS:.*]] = load { float, float }, ptr %[[LHS]], align 4
! CHECK: %[[LOAD_RHS:.*]] = load { float, float }, ptr %[[RHS]], align 4
! CHECK: %[[LHS_REAL:.*]] = extractvalue { float, float } %[[LOAD_LHS]], 0
! CHECK: %[[LHS_IMAG:.*]] = extractvalue { float, float } %[[LOAD_LHS]], 1
! CHECK: %[[RHS_REAL:.*]] = extractvalue { float, float } %[[LOAD_RHS]], 0
! CHECK: %[[RHS_IMAG:.*]] = extractvalue { float, float } %[[LOAD_RHS]], 1

! IMPRVD: %[[RHS_REAL_IMAG_RATIO:.*]] = fdiv contract float %[[RHS_REAL]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract float %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_REAL_IMAG_DENOM:.*]] = fadd contract float %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_1:.*]] = fadd contract float %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_1:.*]] = fdiv contract float %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_1:.*]] = fsub contract float %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_1:.*]] = fdiv contract float %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[RHS_IMAG_REAL_RATIO:.*]] = fdiv contract float %[[RHS_IMAG]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract float %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_IMAG_REAL_DENOM:.*]] = fadd contract float %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_2:.*]] = fadd contract float %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_REAL_2:.*]] = fdiv contract float %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_2:.*]] = fsub contract float %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_IMAG_2:.*]] = fdiv contract float %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]

! Case 1. Zero denominator, numerator contains at most one NaN value.
! IMPRVD: %[[RHS_REAL_ABS:.*]] = call contract float @llvm.fabs.f32(float %[[RHS_REAL]])
! IMPRVD: %[[RHS_REAL_ABS_IS_ZERO:.*]] = fcmp oeq float %[[RHS_REAL_ABS]], 0.000000e+00
! IMPRVD: %[[RHS_IMAG_ABS:.*]] = call contract float @llvm.fabs.f32(float %[[RHS_IMAG]])
! IMPRVD: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = fcmp oeq float %[[RHS_IMAG_ABS]], 0.000000e+00
! IMPRVD: %[[LHS_REAL_IS_NOT_NAN:.*]] = fcmp ord float %[[LHS_REAL]], 0.000000e+00
! IMPRVD: %[[LHS_IMAG_IS_NOT_NAN:.*]] = fcmp ord float %[[LHS_IMAG]], 0.000000e+00
! IMPRVD: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or i1 %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]]
! IMPRVD: %[[RHS_IS_ZERO:.*]] = and i1 %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]]
! IMPRVD: %[[RESULT_IS_INFINITY:.*]] = and i1 %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]]
! IMPRVD: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = call float @llvm.copysign.f32(float 0x7FF0000000000000, float %[[RHS_REAL]])
! IMPRVD: %[[INFINITY_RESULT_REAL:.*]] = fmul contract float %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]]
! IMPRVD: %[[INFINITY_RESULT_IMAG:.*]] = fmul contract float %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]]

! Case 2. Infinite numerator, finite denominator.
! IMPRVD: %[[RHS_REAL_FINITE:.*]] = fcmp one float %[[RHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IMAG_FINITE:.*]] = fcmp one float %[[RHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IS_FINITE:.*]] = and i1 %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]]
! IMPRVD: %[[LHS_REAL_ABS:.*]] = call contract float @llvm.fabs.f32(float %[[LHS_REAL]])
! IMPRVD: %[[LHS_REAL_INFINITE:.*]] = fcmp oeq float %[[LHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IMAG_ABS:.*]] = call contract float @llvm.fabs.f32(float %[[LHS_IMAG]])
! IMPRVD: %[[LHS_IMAG_INFINITE:.*]] = fcmp oeq float %[[LHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IS_INFINITE:.*]] = or i1 %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]]
! IMPRVD: %[[INF_NUM_FINITE_DENOM:.*]] = and i1 %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]]
! IMPRVD: %[[LHS_REAL_IS_INF:.*]] = select i1 %[[LHS_REAL_INFINITE]], float 1.000000e+00, float 0.000000e+00
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = call float @llvm.copysign.f32(float %[[LHS_REAL_IS_INF]], float %[[LHS_REAL]])
! IMPRVD: %[[LHS_IMAG_IS_INF:.*]] = select i1 %[[LHS_IMAG_INFINITE]], float 1.000000e+00, float 0.000000e+00
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = call float @llvm.copysign.f32(float %[[LHS_IMAG_IS_INF]], float %[[LHS_IMAG]])
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract float %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract float %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[INF_MULTIPLICATOR_1:.*]] = fadd contract float %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_3:.*]] = fmul contract float %[[INF_MULTIPLICATOR_1]], 0x7FF0000000000000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract float %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract float %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[INF_MULTIPLICATOR_2:.*]] = fsub contract float %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_IMAG_3:.*]] = fmul contract float %[[INF_MULTIPLICATOR_2]], 0x7FF0000000000000

! Case 3. Finite numerator, infinite denominator.
! IMPRVD: %[[LHS_REAL_FINITE:.*]] = fcmp one float %[[LHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IMAG_FINITE:.*]] = fcmp one float %[[LHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IS_FINITE:.*]] = and i1 %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]]
! IMPRVD: %[[RHS_REAL_INFINITE:.*]] = fcmp oeq float %[[RHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IMAG_INFINITE:.*]] = fcmp oeq float %[[RHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IS_INFINITE:.*]] = or i1 %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]]
! IMPRVD: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and i1 %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]]
! IMPRVD: %[[RHS_REAL_IS_INF:.*]] = select i1 %[[RHS_REAL_INFINITE]], float 1.000000e+00, float 0.000000e+00
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = call float @llvm.copysign.f32(float %[[RHS_REAL_IS_INF]], float %[[RHS_REAL]])
! IMPRVD: %[[RHS_IMAG_IS_INF:.*]] = select i1 %[[RHS_IMAG_INFINITE]], float 1.000000e+00, float 0.000000e+00
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = call float @llvm.copysign.f32(float %[[RHS_IMAG_IS_INF]], float %[[RHS_IMAG]])
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_1:.*]] = fadd contract float %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_4:.*]] = fmul contract float %[[ZERO_MULTIPLICATOR_1]], 0.000000e+00
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_2:.*]] = fsub contract float %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_4:.*]] = fmul contract float %[[ZERO_MULTIPLICATOR_2]], 0.000000e+00

! IMPRVD: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = fcmp olt float %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]]
! IMPRVD: %[[RESULT_REAL:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], float %[[RESULT_REAL_1]], float %[[RESULT_REAL_2]]
! IMPRVD: %[[RESULT_IMAG:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], float %[[RESULT_IMAG_1]], float %[[RESULT_IMAG_2]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], float %[[RESULT_REAL_4]], float %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], float %[[RESULT_IMAG_4]], float %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], float %[[RESULT_REAL_3]], float %[[RESULT_REAL_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], float %[[RESULT_IMAG_3]], float %[[RESULT_IMAG_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], float %[[INFINITY_RESULT_REAL]], float %[[RESULT_REAL_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], float %[[INFINITY_RESULT_IMAG]], float %[[RESULT_IMAG_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_REAL_IS_NAN:.*]] = fcmp uno float %[[RESULT_REAL]], 0.000000e+00
! IMPRVD: %[[RESULT_IMAG_IS_NAN:.*]] = fcmp uno float %[[RESULT_IMAG]], 0.000000e+00
! IMPRVD: %[[RESULT_IS_NAN:.*]] = and i1 %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]]
! IMPRVD: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], float %[[RESULT_REAL_SPECIAL_CASE_1]], float %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], float %[[RESULT_IMAG_SPECIAL_CASE_1]], float %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_1:.*]] = insertvalue { float, float } poison, float %[[RESULT_REAL_WITH_SPECIAL_CASES]], 0
! IMPRVD: %[[RESULT_2:.*]] = insertvalue { float, float } %[[RESULT_1]], float %[[RESULT_IMAG_WITH_SPECIAL_CASES]], 1
! IMPRVD: store { float, float } %[[RESULT_2]], ptr %[[RET]], align 4

! BASIC-DAG: %[[RHS_REAL_SQ:.*]] = fmul contract float %[[RHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[RHS_IMAG_SQ:.*]] = fmul contract float %[[RHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[SQ_NORM:.*]] = fadd contract float %[[RHS_REAL_SQ]], %[[RHS_IMAG_SQ]]
! BASIC-DAG: %[[REAL_TMP_0:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[REAL_TMP_1:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[REAL_TMP_2:.*]] = fadd contract float %[[REAL_TMP_0]], %[[REAL_TMP_1]]
! BASIC-DAG: %[[IMAG_TMP_0:.*]] = fmul contract float %[[LHS_IMAG]], %[[RHS_REAL]]
! BASIC-DAG: %[[IMAG_TMP_1:.*]] = fmul contract float %[[LHS_REAL]], %[[RHS_IMAG]]
! BASIC: %[[IMAG_TMP_2:.*]] = fsub contract float %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]
! BASIC: %[[REAL:.*]] = fdiv contract float %[[REAL_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[IMAG:.*]] = fdiv contract float %[[IMAG_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[RESULT_1:.*]] = insertvalue { float, float } poison, float %[[REAL]], 0
! BASIC: %[[RESULT_2:.*]] = insertvalue { float, float } %[[RESULT_1]], float %[[IMAG]], 1
! BASIC: store { float, float } %[[RESULT_2]], ptr %[[RET]], align 4

! CHECK: ret void
subroutine div_test_single(a,b,c)
  complex(kind=4) :: a, b, c
  a = b / c
end subroutine div_test_single


! CHECK-LABEL: @div_test_double
! CHECK-SAME: ptr noalias %[[RET:.*]], ptr noalias %[[LHS:.*]], ptr noalias %[[RHS:.*]])
! CHECK: %[[LOAD_LHS:.*]] = load { double, double }, ptr %[[LHS]], align 8
! CHECK: %[[LOAD_RHS:.*]] = load { double, double }, ptr %[[RHS]], align 8
! CHECK: %[[LHS_REAL:.*]] = extractvalue { double, double } %[[LOAD_LHS]], 0
! CHECK: %[[LHS_IMAG:.*]] = extractvalue { double, double } %[[LOAD_LHS]], 1
! CHECK: %[[RHS_REAL:.*]] = extractvalue { double, double } %[[LOAD_RHS]], 0
! CHECK: %[[RHS_IMAG:.*]] = extractvalue { double, double } %[[LOAD_RHS]], 1

! IMPRVD: %[[RHS_REAL_IMAG_RATIO:.*]] = fdiv contract double %[[RHS_REAL]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract double %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_REAL_IMAG_DENOM:.*]] = fadd contract double %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_1:.*]] = fadd contract double %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_1:.*]] = fdiv contract double %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_1:.*]] = fsub contract double %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_1:.*]] = fdiv contract double %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]]
! IMPRVD: %[[RHS_IMAG_REAL_RATIO:.*]] = fdiv contract double %[[RHS_IMAG]], %[[RHS_REAL]]
! IMPRVD: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract double %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]]
! IMPRVD: %[[RHS_IMAG_REAL_DENOM:.*]] = fadd contract double %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[REAL_NUMERATOR_2:.*]] = fadd contract double %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_REAL_2:.*]] = fdiv contract double %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]
! IMPRVD: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[IMAG_NUMERATOR_2:.*]] = fsub contract double %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]]
! IMPRVD: %[[RESULT_IMAG_2:.*]] = fdiv contract double %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]]

! Case 1. Zero denominator, numerator contains at most one NaN value.
! IMPRVD: %[[RHS_REAL_ABS:.*]] = call contract double @llvm.fabs.f64(double %[[RHS_REAL]])
! IMPRVD: %[[RHS_REAL_ABS_IS_ZERO:.*]] = fcmp oeq double %[[RHS_REAL_ABS]], 0.000000e+00
! IMPRVD: %[[RHS_IMAG_ABS:.*]] = call contract double @llvm.fabs.f64(double %[[RHS_IMAG]])
! IMPRVD: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = fcmp oeq double %[[RHS_IMAG_ABS]], 0.000000e+00
! IMPRVD: %[[LHS_REAL_IS_NOT_NAN:.*]] = fcmp ord double %[[LHS_REAL]], 0.000000e+00
! IMPRVD: %[[LHS_IMAG_IS_NOT_NAN:.*]] = fcmp ord double %[[LHS_IMAG]], 0.000000e+00
! IMPRVD: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or i1 %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]]
! IMPRVD: %[[RHS_IS_ZERO:.*]] = and i1 %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]]
! IMPRVD: %[[RESULT_IS_INFINITY:.*]] = and i1 %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]]
! IMPRVD: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = call double @llvm.copysign.f64(double 0x7FF0000000000000, double %[[RHS_REAL]])
! IMPRVD: %[[INFINITY_RESULT_REAL:.*]] = fmul contract double %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]]
! IMPRVD: %[[INFINITY_RESULT_IMAG:.*]] = fmul contract double %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]]

! Case 2. Infinite numerator, finite denominator.
! IMPRVD: %[[RHS_REAL_FINITE:.*]] = fcmp one double %[[RHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IMAG_FINITE:.*]] = fcmp one double %[[RHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IS_FINITE:.*]] = and i1 %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]]
! IMPRVD: %[[LHS_REAL_ABS:.*]] = call contract double @llvm.fabs.f64(double %[[LHS_REAL]])
! IMPRVD: %[[LHS_REAL_INFINITE:.*]] = fcmp oeq double %[[LHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IMAG_ABS:.*]] = call contract double @llvm.fabs.f64(double %[[LHS_IMAG]])
! IMPRVD: %[[LHS_IMAG_INFINITE:.*]] = fcmp oeq double %[[LHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IS_INFINITE:.*]] = or i1 %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]]
! IMPRVD: %[[INF_NUM_FINITE_DENOM:.*]] = and i1 %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]]
! IMPRVD: %[[LHS_REAL_IS_INF:.*]] = select i1 %[[LHS_REAL_INFINITE]], double 1.000000e+00, double 0.000000e+00
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = call double @llvm.copysign.f64(double %[[LHS_REAL_IS_INF]], double %[[LHS_REAL]])
! IMPRVD: %[[LHS_IMAG_IS_INF:.*]] = select i1 %[[LHS_IMAG_INFINITE]], double 1.000000e+00, double 0.000000e+00
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = call double @llvm.copysign.f64(double %[[LHS_IMAG_IS_INF]], double %[[LHS_IMAG]])
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract double %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract double %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[INF_MULTIPLICATOR_1:.*]] = fadd contract double %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_3:.*]] = fmul contract double %[[INF_MULTIPLICATOR_1]], 0x7FF0000000000000
! IMPRVD: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = fmul contract double %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]]
! IMPRVD: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = fmul contract double %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]]
! IMPRVD: %[[INF_MULTIPLICATOR_2:.*]] = fsub contract double %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]]
! IMPRVD: %[[RESULT_IMAG_3:.*]] = fmul contract double %[[INF_MULTIPLICATOR_2]], 0x7FF0000000000000

! Case 3. Finite numerator, infinite denominator.
! IMPRVD: %[[LHS_REAL_FINITE:.*]] = fcmp one double %[[LHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IMAG_FINITE:.*]] = fcmp one double %[[LHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[LHS_IS_FINITE:.*]] = and i1 %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]]
! IMPRVD: %[[RHS_REAL_INFINITE:.*]] = fcmp oeq double %[[RHS_REAL_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IMAG_INFINITE:.*]] = fcmp oeq double %[[RHS_IMAG_ABS]], 0x7FF0000000000000
! IMPRVD: %[[RHS_IS_INFINITE:.*]] = or i1 %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]]
! IMPRVD: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and i1 %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]]
! IMPRVD: %[[RHS_REAL_IS_INF:.*]] = select i1 %[[RHS_REAL_INFINITE]], double 1.000000e+00, double 0.000000e+00
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = call double @llvm.copysign.f64(double %[[RHS_REAL_IS_INF]], double %[[RHS_REAL]])
! IMPRVD: %[[RHS_IMAG_IS_INF:.*]] = select i1 %[[RHS_IMAG_INFINITE]], double 1.000000e+00, double 0.000000e+00
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = call double @llvm.copysign.f64(double %[[RHS_IMAG_IS_INF]], double %[[RHS_IMAG]])
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_1:.*]] = fadd contract double %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]]
! IMPRVD: %[[RESULT_REAL_4:.*]] = fmul contract double %[[ZERO_MULTIPLICATOR_1]], 0.000000e+00
! IMPRVD: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]]
! IMPRVD: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]]
! IMPRVD: %[[ZERO_MULTIPLICATOR_2:.*]] = fsub contract double %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]]
! IMPRVD: %[[RESULT_IMAG_4:.*]] = fmul contract double %[[ZERO_MULTIPLICATOR_2]], 0.000000e+00

! IMPRVD: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = fcmp olt double %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]]
! IMPRVD: %[[RESULT_REAL:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], double %[[RESULT_REAL_1]], double %[[RESULT_REAL_2]]
! IMPRVD: %[[RESULT_IMAG:.*]] = select i1 %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], double %[[RESULT_IMAG_1]], double %[[RESULT_IMAG_2]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], double %[[RESULT_REAL_4]], double %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select i1 %[[FINITE_NUM_INFINITE_DENOM]], double %[[RESULT_IMAG_4]], double %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], double %[[RESULT_REAL_3]], double %[[RESULT_REAL_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select i1 %[[INF_NUM_FINITE_DENOM]], double %[[RESULT_IMAG_3]], double %[[RESULT_IMAG_SPECIAL_CASE_3]]
! IMPRVD: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], double %[[INFINITY_RESULT_REAL]], double %[[RESULT_REAL_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select i1 %[[RESULT_IS_INFINITY]], double %[[INFINITY_RESULT_IMAG]], double %[[RESULT_IMAG_SPECIAL_CASE_2]]
! IMPRVD: %[[RESULT_REAL_IS_NAN:.*]] = fcmp uno double %[[RESULT_REAL]], 0.000000e+00
! IMPRVD: %[[RESULT_IMAG_IS_NAN:.*]] = fcmp uno double %[[RESULT_IMAG]], 0.000000e+00
! IMPRVD: %[[RESULT_IS_NAN:.*]] = and i1 %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]]
! IMPRVD: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], double %[[RESULT_REAL_SPECIAL_CASE_1]], double %[[RESULT_REAL]]
! IMPRVD: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select i1 %[[RESULT_IS_NAN]], double %[[RESULT_IMAG_SPECIAL_CASE_1]], double %[[RESULT_IMAG]]
! IMPRVD: %[[RESULT_1:.*]] = insertvalue { double, double } poison, double %[[RESULT_REAL_WITH_SPECIAL_CASES]], 0
! IMPRVD: %[[RESULT_2:.*]] = insertvalue { double, double } %[[RESULT_1]], double %[[RESULT_IMAG_WITH_SPECIAL_CASES]], 1
! IMPRVD: store { double, double } %[[RESULT_2]], ptr %[[RET]], align 8

! BASIC-DAG: %[[RHS_REAL_SQ:.*]] = fmul contract double %[[RHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[RHS_IMAG_SQ:.*]] = fmul contract double %[[RHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[SQ_NORM:.*]] = fadd contract double %[[RHS_REAL_SQ]], %[[RHS_IMAG_SQ]]
! BASIC-DAG: %[[REAL_TMP_0:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_REAL]]
! BASIC-DAG: %[[REAL_TMP_1:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_IMAG]]
! BASIC: %[[REAL_TMP_2:.*]] = fadd contract double %[[REAL_TMP_0]], %[[REAL_TMP_1]]
! BASIC-DAG: %[[IMAG_TMP_0:.*]] = fmul contract double %[[LHS_IMAG]], %[[RHS_REAL]]
! BASIC-DAG: %[[IMAG_TMP_1:.*]] = fmul contract double %[[LHS_REAL]], %[[RHS_IMAG]]
! BASIC: %[[IMAG_TMP_2:.*]] = fsub contract double %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]
! BASIC: %[[REAL:.*]] = fdiv contract double %[[REAL_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[IMAG:.*]] = fdiv contract double %[[IMAG_TMP_2]], %[[SQ_NORM]]
! BASIC: %[[RESULT_1:.*]] = insertvalue { double, double } poison, double %[[REAL]], 0
! BASIC: %[[RESULT_2:.*]] = insertvalue { double, double } %[[RESULT_1]], double %[[IMAG]], 1
! BASIC: store { double, double } %[[RESULT_2]], ptr %[[RET]], align 8

! CHECK: ret void
subroutine div_test_double(a,b,c)
  complex(kind=8) :: a, b, c
  a = b / c
end subroutine div_test_double
