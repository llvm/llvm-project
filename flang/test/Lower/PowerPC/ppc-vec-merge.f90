! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!------------
! vec_mergeh
!------------

  ! CHECK-LABEL: vec_mergeh_test_i1
subroutine vec_mergeh_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i1

! CHECK-LABEL: vec_mergeh_test_i2
subroutine vec_mergeh_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i2

! CHECK-LABEL: vec_mergeh_test_i4
subroutine vec_mergeh_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i4

! CHECK-LABEL: vec_mergeh_test_i8
subroutine vec_mergeh_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i8

! CHECK-LABEL: vec_mergeh_test_u1
subroutine vec_mergeh_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u1

! CHECK-LABEL: vec_mergeh_test_u2
subroutine vec_mergeh_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u2

! CHECK-LABEL: vec_mergeh_test_u4
subroutine vec_mergeh_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u4

! CHECK-LABEL: vec_mergeh_test_u8
subroutine vec_mergeh_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u8

! CHECK-LABEL: vec_mergeh_test_r4
subroutine vec_mergeh_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_r4

! CHECK-LABEL: vec_mergeh_test_r8
subroutine vec_mergeh_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 2>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_r8

!------------
! vec_mergel
!------------

! CHECK-LABEL: vec_mergel_test_i1
subroutine vec_mergel_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i1

! CHECK-LABEL: vec_mergel_test_i2
subroutine vec_mergel_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i2

! CHECK-LABEL: vec_mergel_test_i4
subroutine vec_mergel_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i4

! CHECK-LABEL: vec_mergel_test_i8
subroutine vec_mergel_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i8

! CHECK-LABEL: vec_mergel_test_u1
subroutine vec_mergel_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
! LLVMIR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u1

! CHECK-LABEL: vec_mergel_test_u2
subroutine vec_mergel_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
! LLVMIR: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u2

! CHECK-LABEL: vec_mergel_test_u4
subroutine vec_mergel_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u4

! CHECK-LABEL: vec_mergel_test_u8
subroutine vec_mergel_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u8

! CHECK-LABEL: vec_mergel_test_r4
subroutine vec_mergel_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! LLVMIR: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_r4

! CHECK-LABEL: vec_mergel_test_r8
subroutine vec_mergel_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 3>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_r8
