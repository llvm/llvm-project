! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_cvf_test_r4r8
subroutine vec_cvf_test_r4r8(arg1)
  vector(real(8)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_cvf(arg1)

! LLVMIR: %[[arg:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[call:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvcvdpsp(<2 x double> %[[arg]])
! LLVMIR: store <4 x float> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r4r8

! CHECK-LABEL: vec_cvf_test_r8r4
subroutine vec_cvf_test_r8r4(arg1)
  vector(real(4)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_cvf(arg1)

! LLVMIR: %[[arg:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvcvspdp(<4 x float> %[[arg]])
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r8r4
