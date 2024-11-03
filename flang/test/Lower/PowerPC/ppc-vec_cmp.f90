! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_cmpge
!----------------------

! CHECK-LABEL: vec_cmpge_test_i8
subroutine vec_cmpge_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i64
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i64 to vector<2xi64>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsd(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:ui64>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i64) : i64
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsd(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmpge_test_i8

! CHECK-LABEL: vec_cmpge_test_i4
subroutine vec_cmpge_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i32
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i32 to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsw(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:ui32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i32) : i32
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsw(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmpge_test_i4

! CHECK-LABEL: vec_cmpge_test_i2
subroutine vec_cmpge_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i16
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i16 to vector<8xi16>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsh(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:ui16>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i16) : i16
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsh(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmpge_test_i2

! CHECK-LABEL: vec_cmpge_test_i1
subroutine vec_cmpge_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsb(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:ui8>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsb(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmpge_test_i1

! CHECK-LABEL: vec_cmpge_test_u8
subroutine vec_cmpge_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i64
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i64 to vector<2xi64>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtud(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i64) : i64
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtud(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmpge_test_u8

! CHECK-LABEL: vec_cmpge_test_u4
subroutine vec_cmpge_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i32
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i32 to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtuw(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i32) : i32
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtuw(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmpge_test_u4

! CHECK-LABEL: vec_cmpge_test_u2
subroutine vec_cmpge_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i16
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i16 to vector<8xi16>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtuh(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i16) : i16
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtuh(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmpge_test_u2

! CHECK-LABEL: vec_cmpge_test_u1
subroutine vec_cmpge_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtub(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtub(%[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmpge_test_u1

subroutine vec_cmpge_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgesp(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_cmpge_test_r4

subroutine vec_cmpge_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgedp(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_cmpge_test_r8

!----------------------
! vec_cmpgt
!----------------------

! CHECK-LABEL: vec_cmpgt_test_i1
subroutine vec_cmpgt_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsb(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:ui8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
end subroutine vec_cmpgt_test_i1

! CHECK-LABEL: vec_cmpgt_test_i2
subroutine vec_cmpgt_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsh(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:ui16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
end subroutine vec_cmpgt_test_i2

! CHECK-LABEL: vec_cmpgt_test_i4
subroutine vec_cmpgt_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsw(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_cmpgt_test_i4

! CHECK-LABEL: vec_cmpgt_test_i8
subroutine vec_cmpgt_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsd(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
end subroutine vec_cmpgt_test_i8

! CHECK-LABEL: vec_cmpgt_test_u1
subroutine vec_cmpgt_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtub(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
end subroutine vec_cmpgt_test_u1

! CHECK-LABEL: vec_cmpgt_test_u2
subroutine vec_cmpgt_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuh(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
end subroutine vec_cmpgt_test_u2

! CHECK-LABEL: vec_cmpgt_test_u4
subroutine vec_cmpgt_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuw(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_cmpgt_test_u4

! CHECK-LABEL: vec_cmpgt_test_u8
subroutine vec_cmpgt_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtud(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
end subroutine vec_cmpgt_test_u8

! CHECK-LABEL: vec_cmpgt_test_r4
subroutine vec_cmpgt_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgtsp(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_cmpgt_test_r4

! CHECK-LABEL: vec_cmpgt_test_r8
subroutine vec_cmpgt_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmpgt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgtdp(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_cmpgt_test_r8

!----------------------
! vec_cmple
!----------------------

! CHECK-LABEL: vec_cmple_test_i8
subroutine vec_cmple_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i64
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i64 to vector<2xi64>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsd(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:ui64>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i64) : i64
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsd(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmple_test_i8

! CHECK-LABEL: vec_cmple_test_i4
subroutine vec_cmple_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i32
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i32 to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsw(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:ui32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i32) : i32
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsw(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmple_test_i4

! CHECK-LABEL: vec_cmple_test_i2
subroutine vec_cmple_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i16
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i16 to vector<8xi16>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsh(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:ui16>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i16) : i16
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsh(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmple_test_i2

! CHECK-LABEL: vec_cmple_test_i1
subroutine vec_cmple_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtsb(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:ui8>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtsb(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmple_test_i1

! CHECK-LABEL: vec_cmple_test_u8
subroutine vec_cmple_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i64
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i64 to vector<2xi64>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtud(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i64) : i64
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtud(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg1]], <2 x i64> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[res]], <i64 -1, i64 -1>
end subroutine vec_cmple_test_u8

! CHECK-LABEL: vec_cmple_test_u4
subroutine vec_cmple_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i32
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i32 to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtuw(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i32) : i32
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtuw(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[res]], <i32 -1, i32 -1, i32 -1, i32 -1>
end subroutine vec_cmple_test_u4

! CHECK-LABEL: vec_cmple_test_u2
subroutine vec_cmple_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i16
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i16 to vector<8xi16>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtuh(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i16) : i16
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtuh(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg1]], <8 x i16> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[res]], <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
end subroutine vec_cmple_test_u2

! CHECK-LABEL: vec_cmple_test_u1
subroutine vec_cmple_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[c:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vc:.*]] = vector.broadcast %[[c]] : i8 to vector<16xi8>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vcmpgtub(%[[arg1]], %[[arg2]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[xorres:.*]] = arith.xori %[[vres]], %[[vc]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[xorres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[c:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vc:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vcmpgtub(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[res]], %[[vc]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg1]], <16 x i8> %[[arg2]])
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[res]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
end subroutine vec_cmple_test_u1

! CHECK-LABEL: vec_cmple_test_r4
subroutine vec_cmple_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgesp(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %[[arg2]], <4 x float> %[[arg1]])
end subroutine vec_cmple_test_r4

! CHECK-LABEL: vec_cmple_test_r8
subroutine vec_cmple_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmple(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgedp(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %[[arg2]], <2 x double> %[[arg1]])
end subroutine vec_cmple_test_r8

!----------------------
! vec_cmplt
!----------------------

! CHECK-LABEL: vec_cmplt_test_i1
subroutine vec_cmplt_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsb(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:ui8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtsb(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_cmplt_test_i1

! CHECK-LABEL: vec_cmplt_test_i2
subroutine vec_cmplt_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsh(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:ui16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtsh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_cmplt_test_i2

! CHECK-LABEL: vec_cmplt_test_i4
subroutine vec_cmplt_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsw(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtsw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_cmplt_test_i4

! CHECK-LABEL: vec_cmplt_test_i8
subroutine vec_cmplt_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsd(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_cmplt_test_i8

! CHECK-LABEL: vec_cmplt_test_u1
subroutine vec_cmplt_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2
  vector(unsigned(1)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtub(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ppc.altivec.vcmpgtub(<16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_cmplt_test_u1

! CHECK-LABEL: vec_cmplt_test_u2
subroutine vec_cmplt_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2
  vector(unsigned(2)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuh(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ppc.altivec.vcmpgtuh(<8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_cmplt_test_u2

! CHECK-LABEL: vec_cmplt_test_u4
subroutine vec_cmplt_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuw(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vcmpgtuw(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_cmplt_test_u4

! CHECK-LABEL: vec_cmplt_test_u8
subroutine vec_cmplt_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtud(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_cmplt_test_u8

! CHECK-LABEL: vec_cmplt_test_r4
subroutine vec_cmplt_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2
  vector(unsigned(4)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgtsp(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:ui32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %[[arg2]], <4 x float> %[[arg1]])
end subroutine vec_cmplt_test_r4

! CHECK-LABEL: vec_cmplt_test_r8
subroutine vec_cmplt_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2
  vector(unsigned(8)) :: r
  r = vec_cmplt(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgtdp(%[[arg2]], %[[arg1]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:ui64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %[[arg2]], <2 x double> %[[arg1]])
end subroutine vec_cmplt_test_r8

