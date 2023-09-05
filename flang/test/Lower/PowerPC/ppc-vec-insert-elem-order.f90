! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!CHECK-LABEL: vec_insert_testf32i64
subroutine vec_insert_testf32i64(v, x, i8)
  real(4) :: v
  vector(real(4)) :: x
  vector(real(4)) :: r
  integer(8) :: i8
  r = vec_insert(v, x, i8)

! FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f32>
! FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i64
! FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! FIR: %[[c3:.*]] = arith.constant 3 : i64
! FIR: %[[sub:.*]] = llvm.sub %[[c3]], %[[urem]] : i64
! FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[sub]] : i64] : vector<4xf32>
! FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[urem:.*]] = urem i64 %[[i8]], 4
! LLVMIR: %[[sub:.*]] = sub i64 3, %[[urem]]
! LLVMIR: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i64 %[[sub]]
! LLVMIR: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf32i64

!CHECK-LABEL: vec_insert_testi64i8
subroutine vec_insert_testi64i8(v, x, i1, i2, i4, i8)
  integer(8) :: v
  vector(integer(8)) :: x
  vector(integer(8)) :: r
  integer(1) :: i1
  r = vec_insert(v, x, i1)

! FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i8
! FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! FIR: %[[c1:.*]] = arith.constant 1 : i8
! FIR: %[[sub:.*]] = llvm.sub %[[c1]], %[[urem]] : i8
! FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[sub]] : i8] : vector<2xi64>
! FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[urem:.*]] = urem i8 %[[i1]], 2
! LLVMIR: %[[sub:.*]] = sub i8 1, %[[urem]]
! LLVMIR: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i8 %[[sub]]
! LLVMIR: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi64i8
