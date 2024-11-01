! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_abs
!----------------------

! CHECK-LABEL: vec_abs_i1
subroutine vec_abs_i1(arg1)
  vector(integer(1)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[zero:.*]] = arith.constant 0 : i8
! CHECK-FIR: %[[vzero:.*]] = vector.broadcast %[[zero]] : i8 to vector<16xi8>
! CHECK-FIR: %[[sub:.*]] = arith.subi %[[vzero]], %[[varg1]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vmaxsb(%[[sub]], %[[varg1]]) fastmath<contract> : (vector<16xi8>, vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{.*}} = llvm.mlir.constant(0 : i8) : i8
! CHECK-LLVMIR: %[[vzero:.*]] = llvm.mlir.constant(dense<0> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[sub:.*]] = llvm.sub %[[vzero]], %[[arg1]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vmaxsb(%[[sub]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[sub:.*]] = sub <16 x i8> zeroinitializer, %[[arg1]]
! CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %[[sub]], <16 x i8> %[[arg1]])
end subroutine vec_abs_i1

! CHECK-LABEL: vec_abs_i2
subroutine vec_abs_i2(arg1)
  vector(integer(2)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[zero:.*]] = arith.constant 0 : i16
! CHECK-FIR: %[[vzero:.*]] = vector.broadcast %[[zero]] : i16 to vector<8xi16>
! CHECK-FIR: %[[sub:.*]] = arith.subi %[[vzero]], %[[varg1]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vmaxsh(%[[sub]], %[[varg1]]) fastmath<contract> : (vector<8xi16>, vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{.*}} = llvm.mlir.constant(0 : i16) : i16
! CHECK-LLVMIR: %[[vzero:.*]] = llvm.mlir.constant(dense<0> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[sub:.*]] = llvm.sub %[[vzero]], %[[arg1]]  : vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vmaxsh(%[[sub]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[sub:.*]] = sub <8 x i16> zeroinitializer, %[[arg1]]
! CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %[[sub]], <8 x i16> %[[arg1]])
end subroutine vec_abs_i2

! CHECK-LABEL: vec_abs_i4
subroutine vec_abs_i4(arg1)
  vector(integer(4)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[zero:.*]] = arith.constant 0 : i32
! CHECK-FIR: %[[vzero:.*]] = vector.broadcast %[[zero]] : i32 to vector<4xi32>
! CHECK-FIR: %[[sub:.*]] = arith.subi %[[vzero]], %[[varg1]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vmaxsw(%[[sub]], %[[varg1]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{.*}} = llvm.mlir.constant(0 : i32) : i32
! CHECK-LLVMIR: %[[vzero:.*]] = llvm.mlir.constant(dense<0> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[sub:.*]] = llvm.sub %[[vzero]], %[[arg1]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vmaxsw(%[[sub]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[sub:.*]] = sub <4 x i32> zeroinitializer, %[[arg1]]
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %[[sub]], <4 x i32> %[[arg1]])
end subroutine vec_abs_i4

! CHECK-LABEL: vec_abs_i8
subroutine vec_abs_i8(arg1)
  vector(integer(8)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[zero:.*]] = arith.constant 0 : i64
! CHECK-FIR: %[[vzero:.*]] = vector.broadcast %[[zero]] : i64 to vector<2xi64>
! CHECK-FIR: %[[sub:.*]] = arith.subi %[[vzero]], %[[varg1]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vmaxsd(%[[sub]], %[[varg1]]) fastmath<contract> : (vector<2xi64>, vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
! CHECK-LLVMIR: %[[vzero:.*]] = llvm.mlir.constant(dense<0> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[sub:.*]] = llvm.sub %[[vzero]], %[[arg1]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vmaxsd(%[[sub]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[sub:.*]] = sub <2 x i64> zeroinitializer, %[[arg1]]
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %[[sub]], <2 x i64> %[[arg1]])
end subroutine vec_abs_i8

! CHECK-LABEL: vec_abs_r4
subroutine vec_abs_r4(arg1)
  vector(real(4)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.fabs.v4f32(%[[arg1]]) fastmath<contract> : (!fir.vector<4:f32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.fabs.v4f32(%[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>) -> vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call contract <4 x float> @llvm.fabs.v4f32(<4 x float> %[[arg1]])
end subroutine vec_abs_r4

! CHECK-LABEL: vec_abs_r8
subroutine vec_abs_r8(arg1)
  vector(real(8)) :: arg1, r
  r = vec_abs(arg1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.fabs.v2f64(%[[arg1]]) fastmath<contract> : (!fir.vector<2:f64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.fabs.v2f64(%[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>) -> vector<2xf64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call contract <2 x double> @llvm.fabs.v2f64(<2 x double> %[[arg1]])
end subroutine vec_abs_r8

