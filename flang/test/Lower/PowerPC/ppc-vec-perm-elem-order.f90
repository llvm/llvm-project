! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknwon-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------
! vec_perm
!----------------

! CHECK-LABEL: vec_perm_test_i1
subroutine vec_perm_test_i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<16xi8> to vector<4xi32>
! FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<16xi8> to vector<4xi32>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg1]], %[[barg2]], %[[carg3]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! FIR: %[[vcall:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[bcall:.*]] = llvm.bitcast %[[vcall]] : vector<4xi32> to vector<16xi8>
! FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

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

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 0] : vector<2xi64>, vector<2xi64>
! FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 3, i32 0>
! LLVMIR: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i2
