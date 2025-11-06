! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknwon-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------
! vec_perm
!----------------

! CHECK-LABEL: vec_perm_test_i1
subroutine vec_perm_test_i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[barg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! LLVMIR: %[[barg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg1]], <4 x i32> %[[barg2]], <16 x i8> %[[arg3]])
! LLVMIR: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i1

!----------------
! vec_permi
!----------------

! CHECK-LABEL: vec_permi_test_i8i2
subroutine vec_permi_test_i8i2(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 3, i32 0>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i2
