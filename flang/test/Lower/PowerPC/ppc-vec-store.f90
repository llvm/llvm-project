! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_st
!----------------------

! CHECK-LABEL: vec_st_vi1i2vi1
subroutine vec_st_vi1i2vi1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: %[[bcArg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vi1i2vi1

! CHECK-LABEL: vec_st_vi2i2vi2
subroutine vec_st_vi2i2vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: %[[bcArg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vi2i2vi2

! CHECK-LABEL: vec_st_vi4i2vi4
subroutine vec_st_vi4i2vi4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_st_vi4i2vi4

! CHECK-LABEL: vec_st_vu1i4vu1
subroutine vec_st_vu1i4vu1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: %[[bcArg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vu1i4vu1

! CHECK-LABEL: vec_st_vu2i4vu2
subroutine vec_st_vu2i4vu2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: %[[bcArg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vu2i4vu2

! CHECK-LABEL: vec_st_vu4i4vu4
subroutine vec_st_vu4i4vu4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_st_vu4i4vu4

! CHECK-LABEL: vec_st_vi4i4via4
subroutine vec_st_vi4i4via4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1, arg3(5)
  integer(4) :: arg2, i
  call vec_st(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[iextsub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[iextmul:.*]] = mul nsw i64 %[[iextsub]], 1
! LLVMIR: %[[iextmul2:.*]] = mul nsw i64 %[[iextmul]], 1
! LLVMIR: %[[iextadd:.*]] = add nsw i64 %[[iextmul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iextadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[arg1]], ptr %[[gep2]])
end subroutine vec_st_vi4i4via4

!----------------------
! vec_ste
!----------------------

! CHECK-LABEL: vec_ste_vi1i2i1
subroutine vec_ste_vi1i2i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1
  integer(2) :: arg2
  integer(1) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: call void @llvm.ppc.altivec.stvebx(<16 x i8> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi1i2i1

! CHECK-LABEL: vec_ste_vi2i2i2
subroutine vec_ste_vi2i2i2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(2) :: arg2
  integer(2) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: call void @llvm.ppc.altivec.stvehx(<8 x i16> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi2i2i2

! CHECK-LABEL: vec_ste_vi4i2i4
subroutine vec_ste_vi4i2i4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1
  integer(2) :: arg2
  integer(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi4i2i4

! CHECK-LABEL: vec_ste_vu1i4u1
subroutine vec_ste_vu1i4u1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1
  integer(4) :: arg2
  integer(1) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: call void @llvm.ppc.altivec.stvebx(<16 x i8> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu1i4u1

! CHECK-LABEL: vec_ste_vu2i4u2
subroutine vec_ste_vu2i4u2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1
  integer(4) :: arg2
  integer(2) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: call void @llvm.ppc.altivec.stvehx(<8 x i16> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu2i4u2

! CHECK-LABEL: vec_ste_vu4i4u4
subroutine vec_ste_vu4i4u4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1
  integer(4) :: arg2
  integer(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! LLVMIR: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu4i4u4

! CHECK-LABEL: vec_ste_vr4i4r4
subroutine vec_ste_vr4i4r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(4) :: arg2
  real(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[pos:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[bc:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[bc]], ptr %[[pos]])

end subroutine vec_ste_vr4i4r4

! CHECK-LABEL: vec_ste_vi4i4ia4
subroutine vec_ste_vi4i4ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2, i
  integer(4) :: arg3(5)
  call vec_ste(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[gep2]])
end subroutine vec_ste_vi4i4ia4

!----------------------
! vec_stxv
!----------------------

! CHECK-LABEL: vec_stxv_test_vr4i2r4
subroutine vec_stxv_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_stxv(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! LLVMIR: store <4 x float> %[[arg1]], ptr %[[addr]], align 16
end subroutine vec_stxv_test_vr4i2r4

! CHECK-LABEL: vec_stxv_test_vi4i8ia4
subroutine vec_stxv_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_stxv(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i64 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_stxv_test_vi4i8ia4

! CHECK-LABEL: vec_stxv_test_vi2i4vi2
subroutine vec_stxv_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_stxv(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: store <8 x i16> %[[arg1]], ptr %[[addr]], align 16
end subroutine vec_stxv_test_vi2i4vi2

! CHECK-LABEL: vec_stxv_test_vi4i4vai4
subroutine vec_stxv_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_stxv(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_stxv_test_vi4i4vai4

!----------------------
! vec_xst
!----------------------

! CHECK-LABEL: vec_xst_test_vr4i2r4
subroutine vec_xst_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xst(arg1, arg2, arg3)

  
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! LLVMIR: store <4 x float> %[[arg1]], ptr %[[addr]], align 16
end subroutine vec_xst_test_vr4i2r4

! CHECK-LABEL: vec_xst_test_vi4i8ia4
subroutine vec_xst_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xst(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i64 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_xst_test_vi4i8ia4

! CHECK-LABEL: vec_xst_test_vi2i4vi2
subroutine vec_xst_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xst(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: store <8 x i16> %[[arg1]], ptr %[[addr]], align 16
end subroutine vec_xst_test_vi2i4vi2

! CHECK-LABEL: vec_xst_test_vi4i4vai4
subroutine vec_xst_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xst(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_xst_test_vi4i4vai4

!----------------------
! vec_xst_be
!----------------------

! CHECK-LABEL: vec_xst_be_test_vr4i2r4
subroutine vec_xst_be_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xst_be(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! LLVMIR: %[[shf:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shf]], ptr %[[addr]], align 16
end subroutine vec_xst_be_test_vr4i2r4

! CHECK-LABEL: vec_xst_be_test_vi4i8ia4
subroutine vec_xst_be_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xst_be(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i64 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[src]], ptr %[[gep2]], align 16
end subroutine vec_xst_be_test_vi4i8ia4

! CHECK-LABEL: vec_xst_be_test_vi2i4vi2
subroutine vec_xst_be_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xst_be(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[src]], ptr %[[addr]], align 16
end subroutine vec_xst_be_test_vi2i4vi2

! CHECK-LABEL: vec_xst_be_test_vi4i4vai4
subroutine vec_xst_be_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xst_be(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4 
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[src]], ptr %[[gep2]], align 16
end subroutine vec_xst_be_test_vi4i4vai4

!----------------------
! vec_xstd2
!----------------------

! CHECK-LABEL: vec_xstd2_test_vr4i2r4
subroutine vec_xstd2_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xstd2(arg1, arg2, arg3)

  
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <4 x float> %[[arg1]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[src]], ptr %[[addr]], align 16
end subroutine vec_xstd2_test_vr4i2r4

! CHECK-LABEL: vec_xstd2_test_vi4i8ia4
subroutine vec_xstd2_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i64 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <4 x i32> %[[arg1]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[src]], ptr %[[gep2]], align 16
end subroutine vec_xstd2_test_vi4i8ia4

! CHECK-LABEL: vec_xstd2_test_vi2i4vi2
subroutine vec_xstd2_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xstd2(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <8 x i16> %[[arg1]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[src]], ptr %[[addr]], align 16
end subroutine vec_xstd2_test_vi2i4vi2

! CHECK-LABEL: vec_xstd2_test_vi4i4vai4
subroutine vec_xstd2_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4 
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <4 x i32> %[[arg1]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[src]], ptr %[[gep2]], align 16
end subroutine vec_xstd2_test_vi4i4vai4

!----------------------
! vec_xstw4
!----------------------

! CHECK-LABEL: vec_xstw4_test_vr4i2r4
subroutine vec_xstw4_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xstw4(arg1, arg2, arg3)

  
! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! LLVMIR: store <4 x float> %[[arg1]], ptr %[[addr]], align 16
end subroutine vec_xstw4_test_vr4i2r4

! CHECK-LABEL: vec_xstw4_test_vi4i8ia4
subroutine vec_xstw4_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr i32, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i64 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_xstw4_test_vi4i8ia4

! CHECK-LABEL: vec_xstw4_test_vi2i4vi2
subroutine vec_xstw4_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xstw4(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[src]], ptr %[[addr]], align 16
end subroutine vec_xstw4_test_vi2i4vi2

! CHECK-LABEL: vec_xstw4_test_vi4i4vai4
subroutine vec_xstw4_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! LLVMIR: %[[i:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[i]] to i64
! LLVMIR: %[[isub:.*]] = sub nsw i64 %[[iext]], 1
! LLVMIR: %[[imul1:.*]] = mul nsw i64 %[[isub]], 1
! LLVMIR: %[[imul2:.*]] = mul nsw i64 %[[imul1]], 1
! LLVMIR: %[[iadd:.*]] = add nsw i64 %[[imul2]], 0
! LLVMIR: %[[gep1:.*]] = getelementptr <4 x i32>, ptr %2, i64 %[[iadd]]
! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4 
! LLVMIR: %[[gep2:.*]] = getelementptr i8, ptr %[[gep1]], i32 %[[arg2]]
! LLVMIR: store <4 x i32> %[[arg1]], ptr %[[gep2]], align 16
end subroutine vec_xstw4_test_vi4i4vai4
