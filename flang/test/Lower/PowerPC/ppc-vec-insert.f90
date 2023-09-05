! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

! vec_insert

!CHECK-LABEL: vec_insert_testf32
subroutine vec_insert_testf32(v, x, i1, i2, i4, i8)
  real(4) :: v
  vector(real(4)) :: x
  vector(real(4)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<4xf32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 4
! CHECK: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i8 %[[urem]]
! CHECK: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<4xf32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 4
! CHECK: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i16 %[[urem]]
! CHECK: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<4xf32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 4
! CHECK: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i32 %[[urem]]
! CHECK: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<4xf32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load float, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 4
! CHECK: %[[r:.*]] = insertelement <4 x float> %[[x]], float %[[v]], i64 %[[urem]]
! CHECK: store <4 x float> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf32

!CHECK-LABEL: vec_insert_testf64
subroutine vec_insert_testf64(v, x, i1, i2, i4, i8)
  real(8) :: v
  vector(real(8)) :: x
  vector(real(8)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<2xf64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 2
! CHECK: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i8 %[[urem]]
! CHECK: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<2xf64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 2
! CHECK: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i16 %[[urem]]
! CHECK: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<2xf64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 2
! CHECK: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i32 %[[urem]]
! CHECK: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<f64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<2xf64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<f64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load double, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 2
! CHECK: %[[r:.*]] = insertelement <2 x double> %[[x]], double %[[v]], i64 %[[urem]]
! CHECK: store <2 x double> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testf64

!CHECK-LABEL: vec_insert_testi8
subroutine vec_insert_testi8(v, x, i1, i2, i4, i8)
  integer(1) :: v
  vector(integer(1)) :: x
  vector(integer(1)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<16xi8>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 16
! CHECK: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i8 %[[urem]]
! CHECK: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<16xi8>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 16
! CHECK: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i16 %[[urem]]
! CHECK: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<16xi8>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 16
! CHECK: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i32 %[[urem]]
! CHECK: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<16xi8>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 16
! CHECK: %[[r:.*]] = insertelement <16 x i8> %[[x]], i8 %[[v]], i64 %[[urem]]
! CHECK: store <16 x i8> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi8

!CHECK-LABEL: vec_insert_testi16
subroutine vec_insert_testi16(v, x, i1, i2, i4, i8)
  integer(2) :: v
  vector(integer(2)) :: x
  vector(integer(2)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<8xi16>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 8
! CHECK: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i8 %[[urem]]
! CHECK: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<8xi16>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 8
! CHECK: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i16 %[[urem]]
! CHECK: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<8xi16>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 8
! CHECK: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i32 %[[urem]]
! CHECK: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<8xi16>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 8
! CHECK: %[[r:.*]] = insertelement <8 x i16> %[[x]], i16 %[[v]], i64 %[[urem]]
! CHECK: store <8 x i16> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi16

!CHECK-LABEL: vec_insert_testi32
subroutine vec_insert_testi32(v, x, i1, i2, i4, i8)
  integer(4) :: v
  vector(integer(4)) :: x
  vector(integer(4)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<4xi32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 4
! CHECK: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i8 %[[urem]]
! CHECK: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<4xi32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 4
! CHECK: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i16 %[[urem]]
! CHECK: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<4xi32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 4
! CHECK: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i32 %[[urem]]
! CHECK: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<4xi32>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 4
! CHECK: %[[r:.*]] = insertelement <4 x i32> %[[x]], i32 %[[v]], i64 %[[urem]]
! CHECK: store <4 x i32> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi32


!CHECK-LABEL: vec_insert_testi64
subroutine vec_insert_testi64(v, x, i1, i2, i4, i8)
  integer(8) :: v
  vector(integer(8)) :: x
  vector(integer(8)) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_insert(v, x, i1)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i8
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i8] : vector<2xi64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i1]], %[[c]] : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i8] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[urem:.*]] = urem i8 %[[i1]], 2
! CHECK: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i8 %[[urem]]
! CHECK: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16


  r = vec_insert(v, x, i2)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i16
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i16] : vector<2xi64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i2]], %[[c]] : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i16] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[urem:.*]] = urem i16 %[[i2]], 2
! CHECK: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i16 %[[urem]]
! CHECK: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i4)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i32
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i32] : vector<2xi64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i4]], %[[c]] : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i32] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[urem:.*]] = urem i32 %[[i4]], 2
! CHECK: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i32 %[[urem]]
! CHECK: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16

  r = vec_insert(v, x, i8)
! CHECK-FIR: %[[v:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i64
! CHECK-FIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-FIR: %[[r:.*]] = vector.insertelement %[[v]], %[[vr]][%[[urem]] : i64] : vector<2xi64>
! CHECK-FIR: %[[r_conv:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r_conv]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! CHECK-LLVMIR: %[[urem:.*]] = llvm.urem %[[i8]], %[[c]] : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.insertelement %[[v]], %[[x]][%[[urem]] : i64] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[urem:.*]] = urem i64 %[[i8]], 2
! CHECK: %[[r:.*]] = insertelement <2 x i64> %[[x]], i64 %[[v]], i64 %[[urem]]
! CHECK: store <2 x i64> %[[r]], ptr %{{[0-9]}}, align 16
end subroutine vec_insert_testi64
