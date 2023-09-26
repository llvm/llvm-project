! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_st
!----------------------
! CHECK-LABEL: vec_st_test
subroutine vec_st_test(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[bc:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32> 
! LLVMIR: %[[shf:.*]] = shufflevector <4 x i32> %[[bc]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  call void @llvm.ppc.altivec.stvx(<4 x i32> %[[shf]], ptr %[[addr]])
end subroutine vec_st_test

!----------------------
! vec_ste
!----------------------
! CHECK-LABEL: vec_ste_test
subroutine vec_ste_test(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(4) :: arg2
  real(4) :: arg3
  call vec_ste(arg1, arg2, arg3)
  
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[bc:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[shf:.*]] = shufflevector <4 x i32> %[[bc]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[shf]], ptr %[[addr]])
end subroutine vec_ste_test

!----------------------
! vec_xst
!----------------------
! CHECK-LABEL: vec_xst_test
subroutine vec_xst_test(arg1, arg2, arg3)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3
  call vec_xst(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[trg:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <4 x i32> %[[src]], ptr %[[trg]], align 16
end subroutine vec_xst_test

!----------------------
! vec_xstd2
!----------------------
! CHECK-LABEL: vec_xstd2_test
subroutine vec_xstd2_test(arg1, arg2, arg3, i)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  vector(real(4)) :: arg3(*)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x float>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i16 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <4 x float> %[[arg1]] to <2 x i64>
! LLVMIR: %[[shf:.*]] = shufflevector <2 x i64> %[[src]], <2 x i64> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x i64> %[[shf]], ptr %[[gep2]], align 16
end subroutine vec_xstd2_test

!----------------------
! vec_xstw4
!----------------------
! CHECK-LABEL: vec_xstw4_test
subroutine vec_xstw4_test(arg1, arg2, arg3, i)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  vector(real(4)) :: arg3(*)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x float>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i16 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[src]], ptr %[[gep2]], align 16
end subroutine vec_xstw4_test
