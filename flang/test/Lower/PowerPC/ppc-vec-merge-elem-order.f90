! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!-----------------
! vec_mergeh
!-----------------

! CHECK-LABEL: vec_mergeh_test_i4
subroutine vec_mergeh_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 6, i32 2, i32 7, i32 3>
! LLVMIR: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i4

!-----------------
! vec_mergel
!-----------------

! CHECK-LABEL: vec_mergel_test_r8
subroutine vec_mergel_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 2, i32 0>
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_r8
