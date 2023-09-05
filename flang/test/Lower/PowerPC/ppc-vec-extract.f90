! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!-------------
! vec_extract
!-------------
! CHECK-LABEL: vec_extract_testf32
subroutine vec_extract_testf32(x, i1, i2, i4, i8)
  vector(real(4)) :: x
  real(4) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<4xf32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 4
! CHECK: %[[r:.*]] = extractelement <4 x float> %[[x]], i8 %[[u]]
! CHECK: store float %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<4xf32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 4
! CHECK: %[[r:.*]] = extractelement <4 x float> %[[x]], i16 %[[u]]
! CHECK: store float %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<4xf32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 4
! CHECK: %[[r:.*]] = extractelement <4 x float> %[[x]], i32 %[[u]]
! CHECK: store float %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<4xf32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 4
! CHECK: %[[r:.*]] = extractelement <4 x float> %[[x]], i64 %[[u]]
! CHECK: store float %[[r]], ptr %{{[0-9]}}, align 4
end subroutine vec_extract_testf32

! CHECK-LABEL: vec_extract_testf64
subroutine vec_extract_testf64(x, i1, i2, i4, i8)
  vector(real(8)) :: x
  real(8) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<2xf64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 2
! CHECK: %[[r:.*]] = extractelement <2 x double> %[[x]], i8 %[[u]]
! CHECK: store double %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<2xf64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 2
! CHECK: %[[r:.*]] = extractelement <2 x double> %[[x]], i16 %[[u]]
! CHECK: store double %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<2xf64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 2
! CHECK: %[[r:.*]] = extractelement <2 x double> %[[x]], i32 %[[u]]
! CHECK: store double %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<2xf64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<f64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 2
! CHECK: %[[r:.*]] = extractelement <2 x double> %[[x]], i64 %[[u]]
! CHECK: store double %[[r]], ptr %{{[0-9]}}, align 8
end subroutine vec_extract_testf64

! CHECK-LABEL: vec_extract_testi8
subroutine vec_extract_testi8(x, i1, i2, i4, i8)
  vector(integer(1)) :: x
  integer(1) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<16xi8>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 16
! CHECK: %[[r:.*]] = extractelement <16 x i8> %[[x]], i8 %[[u]]
! CHECK: store i8 %[[r]], ptr %{{[0-9]}}, align 1

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<16xi8>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 16
! CHECK: %[[r:.*]] = extractelement <16 x i8> %[[x]], i16 %[[u]]
! CHECK: store i8 %[[r]], ptr %{{[0-9]}}, align 1

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<16xi8>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 16
! CHECK: %[[r:.*]] = extractelement <16 x i8> %[[x]], i32 %[[u]]
! CHECK: store i8 %[[r]], ptr %{{[0-9]}}, align 1

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[c:.*]] = arith.constant 16 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<16xi8>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(16 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 16
! CHECK: %[[r:.*]] = extractelement <16 x i8> %[[x]], i64 %[[u]]
! CHECK: store i8 %[[r]], ptr %{{[0-9]}}, align 1
end subroutine vec_extract_testi8

! CHECK-LABEL: vec_extract_testi16
subroutine vec_extract_testi16(x, i1, i2, i4, i8)
  vector(integer(2)) :: x
  integer(2) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<8xi16>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 8
! CHECK: %[[r:.*]] = extractelement <8 x i16> %[[x]], i8 %[[u]]
! CHECK: store i16 %[[r]], ptr %{{[0-9]}}, align 2

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<8xi16>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 8
! CHECK: %[[r:.*]] = extractelement <8 x i16> %[[x]], i16 %[[u]]
! CHECK: store i16 %[[r]], ptr %{{[0-9]}}, align 2

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<8xi16>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 8
! CHECK: %[[r:.*]] = extractelement <8 x i16> %[[x]], i32 %[[u]]
! CHECK: store i16 %[[r]], ptr %{{[0-9]}}, align 2

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[c:.*]] = arith.constant 8 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<8xi16>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(8 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 8
! CHECK: %[[r:.*]] = extractelement <8 x i16> %[[x]], i64 %[[u]]
! CHECK: store i16 %[[r]], ptr %{{[0-9]}}, align 2
end subroutine vec_extract_testi16

! CHECK-LABEL: vec_extract_testi32
subroutine vec_extract_testi32(x, i1, i2, i4, i8)
  vector(integer(4)) :: x
  integer(4) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<4xi32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 4
! CHECK: %[[r:.*]] = extractelement <4 x i32> %[[x]], i8 %[[u]]
! CHECK: store i32 %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<4xi32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 4
! CHECK: %[[r:.*]] = extractelement <4 x i32> %[[x]], i16 %[[u]]
! CHECK: store i32 %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<4xi32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 4
! CHECK: %[[r:.*]] = extractelement <4 x i32> %[[x]], i32 %[[u]]
! CHECK: store i32 %[[r]], ptr %{{[0-9]}}, align 4

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[c:.*]] = arith.constant 4 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<4xi32>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 4
! CHECK: %[[r:.*]] = extractelement <4 x i32> %[[x]], i64 %[[u]]
! CHECK: store i32 %[[r]], ptr %{{[0-9]}}, align 4
end subroutine vec_extract_testi32

! CHECK-LABEL: vec_extract_testi64
subroutine vec_extract_testi64(x, i1, i2, i4, i8)
  vector(integer(8)) :: x
  integer(8) :: r
  integer(1) :: i1
  integer(2) :: i2
  integer(4) :: i4
  integer(8) :: i8
  r = vec_extract(x, i1)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i1:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i8>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i8
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i8] : vector<2xi64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i1:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i8>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i1]], %[[c]]  : i8
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i1:.*]] = load i8, ptr %{{[0-9]}}, align 1
! CHECK: %[[u:.*]] = urem i8 %[[i1]], 2
! CHECK: %[[r:.*]] = extractelement <2 x i64> %[[x]], i8 %[[u]]
! CHECK: store i64 %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i2)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i2:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i16>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i16
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i16] : vector<2xi64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i2:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i2]], %[[c]]  : i16
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i2:.*]] = load i16, ptr %{{[0-9]}}, align 2
! CHECK: %[[u:.*]] = urem i16 %[[i2]], 2
! CHECK: %[[r:.*]] = extractelement <2 x i64> %[[x]], i16 %[[u]]
! CHECK: store i64 %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i4)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i4:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i32>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i32
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i32] : vector<2xi64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i4:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i4]], %[[c]]  : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i4:.*]] = load i32, ptr %{{[0-9]}}, align 4
! CHECK: %[[u:.*]] = urem i32 %[[i4]], 2
! CHECK: %[[r:.*]] = extractelement <2 x i64> %[[x]], i32 %[[u]]
! CHECK: store i64 %[[r]], ptr %{{[0-9]}}, align 8

  r = vec_extract(x, i8)
! CHECK-FIR: %[[x:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[i8:.*]] = fir.load %arg{{[0-9]}} : !fir.ref<i64>
! CHECK-FIR: %[[vr:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[c:.*]] = arith.constant 2 : i64
! CHECK-FIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-FIR: %[[r:.*]] = vector.extractelement %[[vr]][%[[u]] : i64] : vector<2xi64>
! CHECK-FIR: fir.store %[[r]] to %{{[0-9]}} : !fir.ref<i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[i8:.*]] = llvm.load %arg{{[0-9]}} : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! CHECK-LLVMIR: %[[u:.*]] = llvm.urem %[[i8]], %[[c]]  : i64
! CHECK-LLVMIR: %[[r:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{[0-9]}} : !llvm.ptr<i64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[i8:.*]] = load i64, ptr %{{[0-9]}}, align 8
! CHECK: %[[u:.*]] = urem i64 %[[i8]], 2
! CHECK: %[[r:.*]] = extractelement <2 x i64> %[[x]], i64 %[[u]]
! CHECK: store i64 %[[r]], ptr %{{[0-9]}}, align 8
end subroutine vec_extract_testi64
