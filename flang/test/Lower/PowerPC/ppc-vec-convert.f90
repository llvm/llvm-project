! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!---------
! vec_ctf
!---------
! CHECK-LABEL: vec_ctf_test_i4i1
subroutine vec_ctf_test_i4i1(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i8
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i8) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:i32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i8) : i8
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.sext %[[arg2]] : i8 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i1

! CHECK-LABEL: vec_ctf_test_i4i2
subroutine vec_ctf_test_i4i2(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i16
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i16) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:i32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i16) : i16
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.sext %[[arg2]] : i16 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i2

! CHECK-LABEL: vec_ctf_test_i4i4
subroutine vec_ctf_test_i4i4(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:i32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i32) : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i4

! CHECK-LABEL: vec_ctf_test_i4i8
subroutine vec_ctf_test_i4i8(arg1)
  vector(integer(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i64) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:i32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.trunc %[[arg2]] : i64 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfsx(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i4i8

! CHECK-LABEL: vec_ctf_test_i8i1
subroutine vec_ctf_test_i8i1(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.sitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.sitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i1

! CHECK-LABEL: vec_ctf_test_i8i2
subroutine vec_ctf_test_i8i2(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.sitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.sitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i2

! CHECK-LABEL: vec_ctf_test_i8i4
subroutine vec_ctf_test_i8i4(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.sitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.sitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i4

! CHECK-LABEL: vec_ctf_test_i8i8
subroutine vec_ctf_test_i8i8(arg1)
  vector(integer(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.sitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.sitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = sitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_i8i8

! CHECK-LABEL: vec_ctf_test_u4i1
subroutine vec_ctf_test_u4i1(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i8
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i8) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:ui32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i8) : i8
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.sext %[[arg2]] : i8 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i1

! CHECK-LABEL: vec_ctf_test_u4i2
subroutine vec_ctf_test_u4i2(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i16
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i16) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:ui32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i16) : i16
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.sext %[[arg2]] : i16 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i2

! CHECK-LABEL: vec_ctf_test_u4i4
subroutine vec_ctf_test_u4i4(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:ui32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i32) : i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i4

! CHECK-LABEL: vec_ctf_test_u4i8
subroutine vec_ctf_test_u4i8(arg1)
  vector(unsigned(4)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_ctf(arg1, 1_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (i64) -> i32
! CHECK-FIR: %[[r:.*]] = fir.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) fastmath<contract> : (!fir.vector<4:ui32>, i32) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[carg2:.*]] = llvm.trunc %[[arg2]] : i64 to i32
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.altivec.vcfux(%[[arg1]], %[[carg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, i32) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = call contract <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %[[arg1]], i32 1)
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u4i8

! CHECK-LABEL: vec_ctf_test_u8i1
subroutine vec_ctf_test_u8i1(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.uitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.uitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i1

! CHECK-LABEL: vec_ctf_test_u8i2
subroutine vec_ctf_test_u8i2(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.uitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.uitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i2

! CHECK-LABEL: vec_ctf_test_u8i4
subroutine vec_ctf_test_u8i4(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.uitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.uitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i4

! CHECK-LABEL: vec_ctf_test_u8i8
subroutine vec_ctf_test_u8i8(arg1)
  vector(unsigned(8)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_ctf(arg1, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg:.*]] = llvm.uitofp %[[varg]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[cst:.*]] = arith.constant dense<1.250000e-01> : vector<2xf64>
! CHECK-FIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[carg:.*]] = llvm.uitofp %[[arg1]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: %[[cst:.*]] = llvm.mlir.constant(dense<1.250000e-01> : vector<2xf64>) : vector<2xf64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.fmul %[[carg]], %[[cst]]  : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[carg:.*]] = uitofp <2 x i64> %[[arg1]] to <2 x double>
! CHECK: %[[r:.*]] = fmul <2 x double> %[[carg]], <double 1.250000e-01, double 1.250000e-01>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_ctf_test_u8i8

!-------------
! vec_convert
!-------------
! CHECK-LABEL: vec_convert_test_i1i1
subroutine vec_convert_test_i1i1(v, mold)
  vector(integer(1)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i1

! CHECK-LABEL: vec_convert_test_i1i2
subroutine vec_convert_test_i1i2(v, mold)
  vector(integer(1)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i2

! CHECK-LABEL: vec_convert_test_i1i4
subroutine vec_convert_test_i1i4(v, mold)
  vector(integer(1)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i4

! CHECK-LABEL: vec_convert_test_i1i8
subroutine vec_convert_test_i1i8(v, mold)
  vector(integer(1)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i8

! CHECK-LABEL: vec_convert_test_i1u1
subroutine vec_convert_test_i1u1(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u1

! CHECK-LABEL: vec_convert_test_i1u2
subroutine vec_convert_test_i1u2(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u2

! CHECK-LABEL: vec_convert_test_i1u4
subroutine vec_convert_test_i1u4(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u4

! CHECK-LABEL: vec_convert_test_i1u8
subroutine vec_convert_test_i1u8(v, mold)
  vector(integer(1)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1u8

! CHECK-LABEL: vec_convert_test_i1r4
subroutine vec_convert_test_i1r4(v, mold)
  vector(integer(1)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1r4

! CHECK-LABEL: vec_convert_test_i1r8
subroutine vec_convert_test_i1r8(v, mold)
  vector(integer(1)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1r8

! CHECK-LABEL: vec_convert_test_i2i1
subroutine vec_convert_test_i2i1(v, mold)
  vector(integer(2)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i1

! CHECK-LABEL: vec_convert_test_i2i2
subroutine vec_convert_test_i2i2(v, mold)
  vector(integer(2)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i2

! CHECK-LABEL: vec_convert_test_i2i4
subroutine vec_convert_test_i2i4(v, mold)
  vector(integer(2)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i4

! CHECK-LABEL: vec_convert_test_i2i8
subroutine vec_convert_test_i2i8(v, mold)
  vector(integer(2)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2i8

! CHECK-LABEL: vec_convert_test_i2u1
subroutine vec_convert_test_i2u1(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u1

! CHECK-LABEL: vec_convert_test_i2u2
subroutine vec_convert_test_i2u2(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u2

! CHECK-LABEL: vec_convert_test_i2u4
subroutine vec_convert_test_i2u4(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u4

! CHECK-LABEL: vec_convert_test_i2u8
subroutine vec_convert_test_i2u8(v, mold)
  vector(integer(2)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2u8

! CHECK-LABEL: vec_convert_test_i2r4
subroutine vec_convert_test_i2r4(v, mold)
  vector(integer(2)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2r4

! CHECK-LABEL: vec_convert_test_i2r8
subroutine vec_convert_test_i2r8(v, mold)
  vector(integer(2)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i2r8

! CHECK-LABEL: vec_convert_test_i4i1
subroutine vec_convert_test_i4i1(v, mold)
  vector(integer(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i1

! CHECK-LABEL: vec_convert_test_i4i2
subroutine vec_convert_test_i4i2(v, mold)
  vector(integer(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i2

! CHECK-LABEL: vec_convert_test_i4i4
subroutine vec_convert_test_i4i4(v, mold)
  vector(integer(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i4

! CHECK-LABEL: vec_convert_test_i4i8
subroutine vec_convert_test_i4i8(v, mold)
  vector(integer(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4i8

! CHECK-LABEL: vec_convert_test_i4u1
subroutine vec_convert_test_i4u1(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u1

! CHECK-LABEL: vec_convert_test_i4u2
subroutine vec_convert_test_i4u2(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u2

! CHECK-LABEL: vec_convert_test_i4u4
subroutine vec_convert_test_i4u4(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u4

! CHECK-LABEL: vec_convert_test_i4u8
subroutine vec_convert_test_i4u8(v, mold)
  vector(integer(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4u8

! CHECK-LABEL: vec_convert_test_i4r4
subroutine vec_convert_test_i4r4(v, mold)
  vector(integer(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r4

! CHECK-LABEL: vec_convert_test_i4r8
subroutine vec_convert_test_i4r8(v, mold)
  vector(integer(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r8

! CHECK-LABEL: vec_convert_test_i8i1
subroutine vec_convert_test_i8i1(v, mold)
  vector(integer(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i1

! CHECK-LABEL: vec_convert_test_i8i2
subroutine vec_convert_test_i8i2(v, mold)
  vector(integer(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i2

! CHECK-LABEL: vec_convert_test_i8i4
subroutine vec_convert_test_i8i4(v, mold)
  vector(integer(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i4

! CHECK-LABEL: vec_convert_test_i8i8
subroutine vec_convert_test_i8i8(v, mold)
  vector(integer(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8i8

! CHECK-LABEL: vec_convert_test_i8u1
subroutine vec_convert_test_i8u1(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u1

! CHECK-LABEL: vec_convert_test_i8u2
subroutine vec_convert_test_i8u2(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u2

! CHECK-LABEL: vec_convert_test_i8u4
subroutine vec_convert_test_i8u4(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u4

! CHECK-LABEL: vec_convert_test_i8u8
subroutine vec_convert_test_i8u8(v, mold)
  vector(integer(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8u8

! CHECK-LABEL: vec_convert_test_i8r4
subroutine vec_convert_test_i8r4(v, mold)
  vector(integer(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8r4

! CHECK-LABEL: vec_convert_test_i8r8
subroutine vec_convert_test_i8r8(v, mold)
  vector(integer(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i8r8

! CHECK-LABEL: vec_convert_test_u1i1
subroutine vec_convert_test_u1i1(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i1

! CHECK-LABEL: vec_convert_test_u1i2
subroutine vec_convert_test_u1i2(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i2

! CHECK-LABEL: vec_convert_test_u1i4
subroutine vec_convert_test_u1i4(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i4

! CHECK-LABEL: vec_convert_test_u1i8
subroutine vec_convert_test_u1i8(v, mold)
  vector(unsigned(1)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1i8

! CHECK-LABEL: vec_convert_test_u1u1
subroutine vec_convert_test_u1u1(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u1

! CHECK-LABEL: vec_convert_test_u1u2
subroutine vec_convert_test_u1u2(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u2

! CHECK-LABEL: vec_convert_test_u1u4
subroutine vec_convert_test_u1u4(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u4

! CHECK-LABEL: vec_convert_test_u1u8
subroutine vec_convert_test_u1u8(v, mold)
  vector(unsigned(1)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1u8

! CHECK-LABEL: vec_convert_test_u1r4
subroutine vec_convert_test_u1r4(v, mold)
  vector(unsigned(1)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1r4

! CHECK-LABEL: vec_convert_test_u1r8
subroutine vec_convert_test_u1r8(v, mold)
  vector(unsigned(1)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u1r8

! CHECK-LABEL: vec_convert_test_u2i1
subroutine vec_convert_test_u2i1(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i1

! CHECK-LABEL: vec_convert_test_u2i2
subroutine vec_convert_test_u2i2(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i2

! CHECK-LABEL: vec_convert_test_u2i4
subroutine vec_convert_test_u2i4(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i4

! CHECK-LABEL: vec_convert_test_u2i8
subroutine vec_convert_test_u2i8(v, mold)
  vector(unsigned(2)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2i8

! CHECK-LABEL: vec_convert_test_u2u1
subroutine vec_convert_test_u2u1(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u1

! CHECK-LABEL: vec_convert_test_u2u2
subroutine vec_convert_test_u2u2(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: store <8 x i16> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u2

! CHECK-LABEL: vec_convert_test_u2u4
subroutine vec_convert_test_u2u4(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u4

! CHECK-LABEL: vec_convert_test_u2u8
subroutine vec_convert_test_u2u8(v, mold)
  vector(unsigned(2)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2u8

! CHECK-LABEL: vec_convert_test_u2r4
subroutine vec_convert_test_u2r4(v, mold)
  vector(unsigned(2)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2r4

! CHECK-LABEL: vec_convert_test_u2r8
subroutine vec_convert_test_u2r8(v, mold)
  vector(unsigned(2)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<8xi16> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<8xi16> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <8 x i16> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u2r8

! CHECK-LABEL: vec_convert_test_u4i1
subroutine vec_convert_test_u4i1(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i1

! CHECK-LABEL: vec_convert_test_u4i2
subroutine vec_convert_test_u4i2(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i2

! CHECK-LABEL: vec_convert_test_u4i4
subroutine vec_convert_test_u4i4(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i4

! CHECK-LABEL: vec_convert_test_u4i8
subroutine vec_convert_test_u4i8(v, mold)
  vector(unsigned(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4i8

! CHECK-LABEL: vec_convert_test_u4u1
subroutine vec_convert_test_u4u1(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u1

! CHECK-LABEL: vec_convert_test_u4u2
subroutine vec_convert_test_u4u2(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u2

! CHECK-LABEL: vec_convert_test_u4u4
subroutine vec_convert_test_u4u4(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: store <4 x i32> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u4

! CHECK-LABEL: vec_convert_test_u4u8
subroutine vec_convert_test_u4u8(v, mold)
  vector(unsigned(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4u8

! CHECK-LABEL: vec_convert_test_u4r4
subroutine vec_convert_test_u4r4(v, mold)
  vector(unsigned(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4r4

! CHECK-LABEL: vec_convert_test_u4r8
subroutine vec_convert_test_u4r8(v, mold)
  vector(unsigned(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xi32> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u4r8

! CHECK-LABEL: vec_convert_test_u8i1
subroutine vec_convert_test_u8i1(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i1

! CHECK-LABEL: vec_convert_test_u8i2
subroutine vec_convert_test_u8i2(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i2

! CHECK-LABEL: vec_convert_test_u8i4
subroutine vec_convert_test_u8i4(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i4

! CHECK-LABEL: vec_convert_test_u8i8
subroutine vec_convert_test_u8i8(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i8

! CHECK-LABEL: vec_convert_test_u8u1
subroutine vec_convert_test_u8u1(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u1

! CHECK-LABEL: vec_convert_test_u8u2
subroutine vec_convert_test_u8u2(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u2

! CHECK-LABEL: vec_convert_test_u8u4
subroutine vec_convert_test_u8u4(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u4

! CHECK-LABEL: vec_convert_test_u8u8
subroutine vec_convert_test_u8u8(v, mold)
  vector(unsigned(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: store <2 x i64> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8u8

! CHECK-LABEL: vec_convert_test_u8r4
subroutine vec_convert_test_u8r4(v, mold)
  vector(unsigned(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8r4

! CHECK-LABEL: vec_convert_test_u8r8
subroutine vec_convert_test_u8r8(v, mold)
  vector(unsigned(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8r8

! CHECK-LABEL: vec_convert_test_r4i1
subroutine vec_convert_test_r4i1(v, mold)
  vector(real(4)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i1

! CHECK-LABEL: vec_convert_test_r4i2
subroutine vec_convert_test_r4i2(v, mold)
  vector(real(4)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i2

! CHECK-LABEL: vec_convert_test_r4i4
subroutine vec_convert_test_r4i4(v, mold)
  vector(real(4)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i4

! CHECK-LABEL: vec_convert_test_r4i8
subroutine vec_convert_test_r4i8(v, mold)
  vector(real(4)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4i8

! CHECK-LABEL: vec_convert_test_r4u1
subroutine vec_convert_test_r4u1(v, mold)
  vector(real(4)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u1

! CHECK-LABEL: vec_convert_test_r4u2
subroutine vec_convert_test_r4u2(v, mold)
  vector(real(4)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u2

! CHECK-LABEL: vec_convert_test_r4u4
subroutine vec_convert_test_r4u4(v, mold)
  vector(real(4)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u4

! CHECK-LABEL: vec_convert_test_r4u8
subroutine vec_convert_test_r4u8(v, mold)
  vector(real(4)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4u8

! CHECK-LABEL: vec_convert_test_r4r4
subroutine vec_convert_test_r4r4(v, mold)
  vector(real(4)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: store <4 x float> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4r4

! CHECK-LABEL: vec_convert_test_r4r8
subroutine vec_convert_test_r4r8(v, mold)
  vector(real(4)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <4 x float> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r4r8

! CHECK-LABEL: vec_convert_test_r8i1
subroutine vec_convert_test_r8i1(v, mold)
  vector(real(8)) :: v
  vector(integer(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i1

! CHECK-LABEL: vec_convert_test_r8i2
subroutine vec_convert_test_r8i2(v, mold)
  vector(real(8)) :: v
  vector(integer(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i2

! CHECK-LABEL: vec_convert_test_r8i4
subroutine vec_convert_test_r8i4(v, mold)
  vector(real(8)) :: v
  vector(integer(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i4

! CHECK-LABEL: vec_convert_test_r8i8
subroutine vec_convert_test_r8i8(v, mold)
  vector(real(8)) :: v
  vector(integer(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8i8

! CHECK-LABEL: vec_convert_test_r8u1
subroutine vec_convert_test_r8u1(v, mold)
  vector(real(8)) :: v
  vector(unsigned(1)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <16 x i8>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u1

! CHECK-LABEL: vec_convert_test_r8u2
subroutine vec_convert_test_r8u2(v, mold)
  vector(real(8)) :: v
  vector(unsigned(2)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u2

! CHECK-LABEL: vec_convert_test_r8u4
subroutine vec_convert_test_r8u4(v, mold)
  vector(real(8)) :: v
  vector(unsigned(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x i32>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u4

! CHECK-LABEL: vec_convert_test_r8u8
subroutine vec_convert_test_r8u8(v, mold)
  vector(real(8)) :: v
  vector(unsigned(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <2 x i64>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8u8

! CHECK-LABEL: vec_convert_test_r8r4
subroutine vec_convert_test_r8r4(v, mold)
  vector(real(8)) :: v
  vector(real(4)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xf64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: %[[r:.*]] = bitcast <2 x double> %[[v]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8r4

! CHECK-LABEL: vec_convert_test_r8r8
subroutine vec_convert_test_r8r8(v, mold)
  vector(real(8)) :: v
  vector(real(8)) :: mold, r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vc:.*]] = fir.convert %[[v]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[vc]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <2 x double>, ptr %0, align 16
! CHECK: store <2 x double> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_r8r8

! CHECK-LABEL: vec_convert_test_i1i1_array
subroutine vec_convert_test_i1i1_array(v, mold)
  vector(integer(1)) :: v
  vector(integer(1)) :: mold(4, 8), r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[cv:.*]] = fir.convert %[[v]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[cv]] : vector<16xi8> to vector<16xi8>
! CHECK-FIR: %[[r:.*]]  = fir.convert %[[b]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: llvm.store %[[v]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[v:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: store <16 x i8> %[[v]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i1i1_array

! CHECK-LABEL: vec_convert_test_i4r8_array
subroutine vec_convert_test_i4r8_array(v, mold)
  vector(integer(4)) :: v
  vector(real(8)) :: mold(2, 4, 8), r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[cv:.*]] = fir.convert %[[v]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[cv]] : vector<4xi32> to vector<2xf64>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<4xi32> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[v:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = bitcast <4 x i32> %[[v]] to <2 x double>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_i4r8_array

! CHECK-LABEL: vec_convert_test_u8i2_array
subroutine vec_convert_test_u8i2_array(v, mold)
  vector(unsigned(8)) :: v
  vector(integer(2)) :: mold(10), r
  r = vec_convert(v, mold)

! CHECK-FIR: %[[v:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[cv:.*]] = fir.convert %[[v]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[b:.*]] = llvm.bitcast %[[cv]] : vector<2xi64> to vector<8xi16>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[b]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[v:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[v]] : vector<2xi64> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[v:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = bitcast <2 x i64> %[[v]] to <8 x i16>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_convert_test_u8i2_array

!---------
! vec_cvf
!---------
! CHECK-LABEL: vec_cvf_test_r4r8
subroutine vec_cvf_test_r4r8(arg1)
  vector(real(8)), intent(in) :: arg1
  vector(real(4)) :: r
  r = vec_cvf(arg1)

! CHECK-FIR: %[[arg:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg:.*]] = fir.convert %[[arg]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.xvcvdpsp(%[[carg]]) fastmath<contract> : (vector<2xf64>) -> !fir.vector<4:f32>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[call]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bfi:.*]] = llvm.bitcast %[[ccall]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[sh:.*]] = vector.shuffle %[[bfi]], %[[bfi]] [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[bif:.*]] = llvm.bitcast %[[sh]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[r:.*]] = fir.convert %[[bif]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[r]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.vsx.xvcvdpsp(%[[arg]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>) -> vector<4xf32>
! CHECK-LLVMIR: %[[b:.*]] = llvm.bitcast %[[call]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[sh:.*]] = llvm.shufflevector %[[b]], %[[b]] [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.bitcast %[[sh]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[call:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvcvdpsp(<2 x double> %[[arg]])
! CHECK: %[[b:.*]] = bitcast <4 x float> %[[call]] to <16 x i8>
! CHECK: %[[sh:.*]] = shufflevector <16 x i8> %[[b]], <16 x i8> %[[b]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11>
! CHECK: %[[r:.*]] = bitcast <16 x i8> %[[sh]] to <4 x float>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r4r8

! CHECK-LABEL: vec_cvf_test_r8r4
subroutine vec_cvf_test_r8r4(arg1)
  vector(real(4)), intent(in) :: arg1
  vector(real(8)) :: r
  r = vec_cvf(arg1)

! CHECK-FIR: %[[arg:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg:.*]] = fir.convert %[[arg]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bfi:.*]] = llvm.bitcast %[[carg]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[sh:.*]] = vector.shuffle %[[bfi]], %[[bfi]] [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[bif:.*]] = llvm.bitcast %[[sh]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.xvcvspdp(%[[bif]]) fastmath<contract> : (vector<4xf32>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[call]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[bfi:.*]] = llvm.bitcast %[[arg]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[sh:.*]] = llvm.shufflevector %[[bfi]], %[[bfi]] [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<16xi8>
! CHECK-LLVMIR: %[[bif:.*]] = llvm.bitcast %[[sh]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: %[[r:.*]] = llvm.call @llvm.ppc.vsx.xvcvspdp(%[[bif]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[bfi:.*]] = bitcast <4 x float> %[[arg]] to <16 x i8>
! CHECK: %[[sh:.*]] = shufflevector <16 x i8> %[[bfi]], <16 x i8> %[[bfi]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11>
! CHECK: %[[bif:.*]] = bitcast <16 x i8> %[[sh]] to <4 x float>
! CHECK: %[[r:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvcvspdp(<4 x float> %[[bif]])
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_cvf_test_r8r4


