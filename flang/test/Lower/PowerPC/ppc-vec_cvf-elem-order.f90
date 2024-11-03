! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_cvf_test_r4r8
subroutine vec_cvf_test_r4r8(arg1)
  vector(real(8)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_cvf(arg1)

! FIR: %[[arg:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! FIR: %[[carg:.*]] = fir.convert %[[arg]] : (!fir.vector<2:f64>) -> vector<2xf64>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.xvcvdpsp(%[[carg]]) fastmath<contract> : (vector<2xf64>) -> !fir.vector<4:f32>
! FIR: %[[ccall:.*]] = fir.convert %[[call]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[r:.*]] = fir.convert %[[ccall]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[call:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvcvdpsp(<2 x double> %[[arg]])
! LLVMIR: store <4 x float> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r4r8

! CHECK-LABEL: vec_cvf_test_r8r4
subroutine vec_cvf_test_r8r4(arg1)
  vector(real(4)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_cvf(arg1)

! FIR: %[[arg:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[carg:.*]] = fir.convert %[[arg]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.xvcvspdp(%[[carg]]) fastmath<contract> : (vector<4xf32>) -> !fir.vector<2:f64>
! FIR: fir.store %[[call]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[arg:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[r:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvcvspdp(<4 x float> %[[arg]])
! LLVMIR: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r8r4
