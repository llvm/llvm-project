! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="CHECK" %s
!
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -triple ppc64-unknown-aix -o - | FileCheck --check-prefixes="BE-LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_sld
!----------------------

! CHECK-LABEL: vec_sld_test_i1i1
subroutine vec_sld_test_i1i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i1

! CHECK-LABEL: vec_sld_test_i1i2
subroutine vec_sld_test_i1i2(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i2

! CHECK-LABEL: vec_sld_test_i1i4
subroutine vec_sld_test_i1i4(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i4

! CHECK-LABEL: vec_sld_test_i1i8
subroutine vec_sld_test_i1i8(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i8

! CHECK-LABEL: vec_sld_test_i2i1
subroutine vec_sld_test_i2i1(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i1

! CHECK-LABEL: vec_sld_test_i2i2
subroutine vec_sld_test_i2i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 8_2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i2

! CHECK-LABEL: vec_sld_test_i2i4
subroutine vec_sld_test_i2i4(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i4

! CHECK-LABEL: vec_sld_test_i2i8
subroutine vec_sld_test_i2i8(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 11_8)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i8

! CHECK-LABEL: vec_sld_test_i4i1
subroutine vec_sld_test_i4i1(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i1

! CHECK-LABEL: vec_sld_test_i4i2
subroutine vec_sld_test_i4i2(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i2

! CHECK-LABEL: vec_sld_test_i4i4
subroutine vec_sld_test_i4i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i4

! CHECK-LABEL: vec_sld_test_i4i8
subroutine vec_sld_test_i4i8(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i8

! CHECK-LABEL: vec_sld_test_u1i1
subroutine vec_sld_test_u1i1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i1

! CHECK-LABEL: vec_sld_test_u1i2
subroutine vec_sld_test_u1i2(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i2

! CHECK-LABEL: vec_sld_test_u1i4
subroutine vec_sld_test_u1i4(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i4

! CHECK-LABEL: vec_sld_test_u1i8
subroutine vec_sld_test_u1i8(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i8

! CHECK-LABEL: vec_sld_test_u2i1
subroutine vec_sld_test_u2i1(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i1

! CHECK-LABEL: vec_sld_test_u2i2
subroutine vec_sld_test_u2i2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i2

! CHECK-LABEL: vec_sld_test_u2i4
subroutine vec_sld_test_u2i4(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i4

! CHECK-LABEL: vec_sld_test_u2i8
subroutine vec_sld_test_u2i8(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i8

! CHECK-LABEL: vec_sld_test_u4i1
subroutine vec_sld_test_u4i1(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i1

! CHECK-LABEL: vec_sld_test_u4i2
subroutine vec_sld_test_u4i2(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i2

! CHECK-LABEL: vec_sld_test_u4i4
subroutine vec_sld_test_u4i4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i4

! CHECK-LABEL: vec_sld_test_u4i8
subroutine vec_sld_test_u4i8(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i8

! CHECK-LABEL: vec_sld_test_r4i1
subroutine vec_sld_test_r4i1(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i1

! CHECK-LABEL: vec_sld_test_r4i2
subroutine vec_sld_test_r4i2(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i2

! CHECK-LABEL: vec_sld_test_r4i4
subroutine vec_sld_test_r4i4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i4

! CHECK-LABEL: vec_sld_test_r4i8
subroutine vec_sld_test_r4i8(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 1_8)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i8

!----------------------
! vec_sldw
!----------------------
! CHECK-LABEL: vec_sldw_test_i1i1
subroutine vec_sldw_test_i1i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i1

! CHECK-LABEL: vec_sldw_test_i1i2
subroutine vec_sldw_test_i1i2(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i2

! CHECK-LABEL: vec_sldw_test_i1i4
subroutine vec_sldw_test_i1i4(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i4

! CHECK-LABEL: vec_sldw_test_i1i8
subroutine vec_sldw_test_i1i8(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i8

! CHECK-LABEL: vec_sldw_test_i2i1
subroutine vec_sldw_test_i2i1(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i1

! CHECK-LABEL: vec_sldw_test_i2i2
subroutine vec_sldw_test_i2i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i2

! CHECK-LABEL: vec_sldw_test_i2i4
subroutine vec_sldw_test_i2i4(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i4

! CHECK-LABEL: vec_sldw_test_i2i8
subroutine vec_sldw_test_i2i8(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i8

! CHECK-LABEL: vec_sldw_test_i4i1
subroutine vec_sldw_test_i4i1(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i1

! CHECK-LABEL: vec_sldw_test_i4i2
subroutine vec_sldw_test_i4i2(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i2

! CHECK-LABEL: vec_sldw_test_i4i4
subroutine vec_sldw_test_i4i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i4

! CHECK-LABEL: vec_sldw_test_i4i8
subroutine vec_sldw_test_i4i8(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i8

! CHECK-LABEL: vec_sldw_test_i8i1
subroutine vec_sldw_test_i8i1(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i1

! CHECK-LABEL: vec_sldw_test_i8i2
subroutine vec_sldw_test_i8i2(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i2

! CHECK-LABEL: vec_sldw_test_i8i4
subroutine vec_sldw_test_i8i4(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i4

! CHECK-LABEL: vec_sldw_test_i8i8
subroutine vec_sldw_test_i8i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

end subroutine vec_sldw_test_i8i8

! CHECK-LABEL: vec_sldw_test_u1i1
subroutine vec_sldw_test_u1i1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i1

! CHECK-LABEL: vec_sldw_test_u1i2
subroutine vec_sldw_test_u1i2(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i2

! CHECK-LABEL: vec_sldw_test_u1i4
subroutine vec_sldw_test_u1i4(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i4

! CHECK-LABEL: vec_sldw_test_u1i8
subroutine vec_sldw_test_u1i8(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i8

! CHECK-LABEL: vec_sldw_test_u2i1
subroutine vec_sldw_test_u2i1(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i1

! CHECK-LABEL: vec_sldw_test_u2i2
subroutine vec_sldw_test_u2i2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i2

! CHECK-LABEL: vec_sldw_test_u2i4
subroutine vec_sldw_test_u2i4(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i4

! CHECK-LABEL: vec_sldw_test_u2i8
subroutine vec_sldw_test_u2i8(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-LLVMIR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i8

! CHECK-LABEL: vec_sldw_test_u4i1
subroutine vec_sldw_test_u4i1(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i1

! CHECK-LABEL: vec_sldw_test_u4i2
subroutine vec_sldw_test_u4i2(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i2

! CHECK-LABEL: vec_sldw_test_u4i4
subroutine vec_sldw_test_u4i4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i4

! CHECK-LABEL: vec_sldw_test_u4i8
subroutine vec_sldw_test_u4i8(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-LLVMIR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i8

! CHECK-LABEL: vec_sldw_test_u8i1
subroutine vec_sldw_test_u8i1(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i1

! CHECK-LABEL: vec_sldw_test_u8i2
subroutine vec_sldw_test_u8i2(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i2

! CHECK-LABEL: vec_sldw_test_u8i4
subroutine vec_sldw_test_u8i4(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i4

! CHECK-LABEL: vec_sldw_test_u8i8
subroutine vec_sldw_test_u8i8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-LLVMIR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i8

! CHECK-LABEL: vec_sldw_test_r4i1
subroutine vec_sldw_test_r4i1(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i1

! CHECK-LABEL: vec_sldw_test_r4i2
subroutine vec_sldw_test_r4i2(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i2

! CHECK-LABEL: vec_sldw_test_r4i4
subroutine vec_sldw_test_r4i4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i4

! CHECK-LABEL: vec_sldw_test_r4i8
subroutine vec_sldw_test_r4i8(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-LLVMIR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i8

! CHECK-LABEL: vec_sldw_test_r8i1
subroutine vec_sldw_test_r8i1(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i1

! CHECK-LABEL: vec_sldw_test_r8i2
subroutine vec_sldw_test_r8i2(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i2

! CHECK-LABEL: vec_sldw_test_r8i4
subroutine vec_sldw_test_r8i4(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i4

! CHECK-LABEL: vec_sldw_test_r8i8
subroutine vec_sldw_test_r8i8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-LLVMIR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-LLVMIR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-LLVMIR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-LLVMIR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i8
