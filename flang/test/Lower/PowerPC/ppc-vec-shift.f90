! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_sl
!----------------------

! CHECK-LABEL: vec_sl_i1
subroutine vec_sl_i1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %c8_i8 = arith.constant 8 : i8
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c8_i8 : i8 to vector<16xi8>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<8> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! CHECK: %7 = shl <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sl_i1

! CHECK-LABEL: vec_sl_i2
subroutine vec_sl_i2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %c16_i16 = arith.constant 16 : i16
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c16_i16 : i16 to vector<8xi16>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<16> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! CHECK: %7 = shl <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sl_i2

! CHECK-LABEL: vec_sl_i4
subroutine vec_sl_i4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %c32_i32 = arith.constant 32 : i32
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c32_i32 : i32 to vector<4xi32>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<32> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! CHECK: %7 = shl <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sl_i4

! CHECK-LABEL: vec_sl_i8
subroutine vec_sl_i8(arg1, arg2)
  vector(integer(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %c64_i64 = arith.constant 64 : i64
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c64_i64 : i64 to vector<2xi64>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<64> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! CHECK: %7 = shl <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sl_i8

! CHECK-LABEL: vec_sl_u1
subroutine vec_sl_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %c8_i8 = arith.constant 8 : i8
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c8_i8 : i8 to vector<16xi8>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<8> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! CHECK: %7 = shl <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sl_u1

! CHECK-LABEL: vec_sl_u2
subroutine vec_sl_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %c16_i16 = arith.constant 16 : i16
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c16_i16 : i16 to vector<8xi16>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<16> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! CHECK: %7 = shl <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sl_u2

! CHECK-LABEL: vec_sl_u4
subroutine vec_sl_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %c32_i32 = arith.constant 32 : i32
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c32_i32 : i32 to vector<4xi32>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<32> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! CHECK: %7 = shl <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sl_u4

! CHECK-LABEL: vec_sl_u8
subroutine vec_sl_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %c64_i64 = arith.constant 64 : i64
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c64_i64 : i64 to vector<2xi64>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.shli %[[varg1]], %[[msk]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<64> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.shl %[[arg1]], %[[msk]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! CHECK: %{{[0-9]+}} = shl <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sl_u8

!----------------------
! vec_sll
!----------------------
! CHECK-LABEL: vec_sll_i1u1
subroutine vec_sll_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u1

! CHECK-LABEL: vec_sll_i2u1
subroutine vec_sll_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u1

! CHECK-LABEL: vec_sll_i4u1
subroutine vec_sll_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_i4u1

! CHECK-LABEL: vec_sll_i1u2
subroutine vec_sll_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u2

! CHECK-LABEL: vec_sll_i2u2
subroutine vec_sll_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u2

! CHECK-LABEL: vec_sll_i4u2
subroutine vec_sll_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_i4u2

! CHECK-LABEL: vec_sll_i1u4
subroutine vec_sll_i1u4(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_i1u4

! CHECK-LABEL: vec_sll_i2u4
subroutine vec_sll_i2u4(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_i2u4

! CHECK-LABEL: vec_sll_i4u4
subroutine vec_sll_i4u4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_sll_i4u4

! CHECK-LABEL: vec_sll_u1u1
subroutine vec_sll_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u1

! CHECK-LABEL: vec_sll_u2u1
subroutine vec_sll_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u1

! CHECK-LABEL: vec_sll_u4u1
subroutine vec_sll_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_u4u1

! CHECK-LABEL: vec_sll_u1u2
subroutine vec_sll_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u2

! CHECK-LABEL: vec_sll_u2u2
subroutine vec_sll_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u2

! CHECK-LABEL: vec_sll_u4u2
subroutine vec_sll_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sll_u4u2

! CHECK-LABEL: vec_sll_u1u4
subroutine vec_sll_u1u4(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sll_u1u4

! CHECK-LABEL: vec_sll_u2u4
subroutine vec_sll_u2u4(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sll_u2u4

! CHECK-LABEL: vec_sll_u4u4
subroutine vec_sll_u4u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sll(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsl(%[[varg1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsl(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsl(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_sll_u4u4

!----------------------
! vec_slo
!----------------------

! CHECK-LABEL: vec_slo_i1u1
subroutine vec_slo_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_i1u1

! CHECK-LABEL: vec_slo_i2u1
subroutine vec_slo_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_i2u1

! CHECK-LABEL: vec_slo_i4u1
subroutine vec_slo_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vslo(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_i4u1

! CHECK-LABEL: vec_slo_u1u1
subroutine vec_slo_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_u1u1

! CHECK-LABEL: vec_slo_u2u1
subroutine vec_slo_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_u2u1

! CHECK-LABEL: vec_slo_u4u1
subroutine vec_slo_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vslo(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_u4u1

! CHECK-LABEL: vec_slo_r4u1
subroutine vec_slo_r4u1(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_slo_r4u1

! CHECK-LABEL: vec_slo_i1u2
subroutine vec_slo_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_i1u2

! CHECK-LABEL: vec_slo_i2u2
subroutine vec_slo_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_slo_i2u2

! CHECK-LABEL: vec_slo_i4u2
subroutine vec_slo_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vslo(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_i4u2

! CHECK-LABEL: vec_slo_u1u2
subroutine vec_slo_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_slo_u1u2

! CHECK-LABEL: vec_slo_u2u2
subroutine vec_slo_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>

end subroutine vec_slo_u2u2

! CHECK-LABEL: vec_slo_u4u2
subroutine vec_slo_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vslo(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_slo_u4u2

! CHECK-LABEL: vec_slo_r4u2
subroutine vec_slo_r4u2(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_slo(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vslo(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vslo(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vslo(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_slo_r4u2

!----------------------
! vec_sr
!----------------------
! CHECK-LABEL: vec_sr_i1
subroutine vec_sr_i1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %c8_i8 = arith.constant 8 : i8
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c8_i8 : i8 to vector<16xi8>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<8> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! CHECK: %7 = lshr <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sr_i1

! CHECK-LABEL: vec_sr_i2
subroutine vec_sr_i2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %c16_i16 = arith.constant 16 : i16
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c16_i16 : i16 to vector<8xi16>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<16> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! CHECK: %7 = lshr <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sr_i2

! CHECK-LABEL: vec_sr_i4
subroutine vec_sr_i4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %c32_i32 = arith.constant 32 : i32
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c32_i32 : i32 to vector<4xi32>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<32> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! CHECK: %7 = lshr <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sr_i4

! CHECK-LABEL: vec_sr_i8
subroutine vec_sr_i8(arg1, arg2)
  vector(integer(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %c64_i64 = arith.constant 64 : i64
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c64_i64 : i64 to vector<2xi64>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<64> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! CHECK: %7 = lshr <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sr_i8

! CHECK-LABEL: vec_sr_u1
subroutine vec_sr_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %c8_i8 = arith.constant 8 : i8
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c8_i8 : i8 to vector<16xi8>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<8> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<16xi8>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <16 x i8> %[[arg2]], <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
! CHECK: %7 = lshr <16 x i8> %[[arg1]], %[[msk]]
end subroutine vec_sr_u1

! CHECK-LABEL: vec_sr_u2
subroutine vec_sr_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %c16_i16 = arith.constant 16 : i16
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c16_i16 : i16 to vector<8xi16>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<16> : vector<8xi16>) : vector<8xi16>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<8xi16>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <8 x i16> %[[arg2]], <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
! CHECK: %7 = lshr <8 x i16> %[[arg1]], %[[msk]]
end subroutine vec_sr_u2

! CHECK-LABEL: vec_sr_u4
subroutine vec_sr_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %c32_i32 = arith.constant 32 : i32
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c32_i32 : i32 to vector<4xi32>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<32> : vector<4xi32>) : vector<4xi32>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <4 x i32> %[[arg2]], <i32 32, i32 32, i32 32, i32 32>
! CHECK: %7 = lshr <4 x i32> %[[arg1]], %[[msk]]
end subroutine vec_sr_u4

! CHECK-LABEL: vec_sr_u8
subroutine vec_sr_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, r
  vector(unsigned(8)) :: arg2
  r = vec_sr(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %c64_i64 = arith.constant 64 : i64
! CHECK-FIR: %[[cv:.*]] = vector.broadcast %c64_i64 : i64 to vector<2xi64>
! CHECK-FIR: %[[msk:.*]] = arith.remui %[[varg2]], %[[cv]] : vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.shrui %[[varg1]], %[[msk]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[cv:.*]] = llvm.mlir.constant(dense<64> : vector<2xi64>) : vector<2xi64>
! CHECK-LLVMIR: %[[msk:.*]] = llvm.urem %[[arg2]], %[[cv]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.lshr %[[arg1]], %[[msk]]  : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[msk:.*]] = urem <2 x i64> %[[arg2]], <i64 64, i64 64>
! CHECK: %7 = lshr <2 x i64> %[[arg1]], %[[msk]]
end subroutine vec_sr_u8

!----------------------
! vec_srl
!----------------------
! CHECK-LABEL: vec_srl_i1u1
subroutine vec_srl_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u1

! CHECK-LABEL: vec_srl_i2u1
subroutine vec_srl_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u1

! CHECK-LABEL: vec_srl_i4u1
subroutine vec_srl_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_i4u1

! CHECK-LABEL: vec_srl_i1u2
subroutine vec_srl_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u2

! CHECK-LABEL: vec_srl_i2u2
subroutine vec_srl_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u2

! CHECK-LABEL: vec_srl_i4u2
subroutine vec_srl_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_i4u2

! CHECK-LABEL: vec_srl_i1u4
subroutine vec_srl_i1u4(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_i1u4

! CHECK-LABEL: vec_srl_i2u4
subroutine vec_srl_i2u4(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_i2u4

! CHECK-LABEL: vec_srl_i4u4
subroutine vec_srl_i4u4(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_srl_i4u4

! CHECK-LABEL: vec_srl_u1u1
subroutine vec_srl_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u1

! CHECK-LABEL: vec_srl_u2u1
subroutine vec_srl_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u1

! CHECK-LABEL: vec_srl_u4u1
subroutine vec_srl_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_u4u1

! CHECK-LABEL: vec_srl_u1u2
subroutine vec_srl_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u2

! CHECK-LABEL: vec_srl_u2u2
subroutine vec_srl_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u2

! CHECK-LABEL: vec_srl_u4u2
subroutine vec_srl_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_srl_u4u2

! CHECK-LABEL: vec_srl_u1u4
subroutine vec_srl_u1u4(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR:    %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_srl_u1u4

! CHECK-LABEL: vec_srl_u2u4
subroutine vec_srl_u2u4(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[bc1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[varg1]], <4 x i32> %[[arg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_srl_u2u4

! CHECK-LABEL: vec_srl_u4u4
subroutine vec_srl_u4u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(4)) :: arg2
  r = vec_srl(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsr(%[[varg1]], %[[varg2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsr(%[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsr(<4 x i32> %[[arg1]], <4 x i32> %[[arg2]])
end subroutine vec_srl_u4u4

!----------------------
! vec_sro
!----------------------

! CHECK-LABEL: vec_sro_i1u1
subroutine vec_sro_i1u1(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_i1u1

! CHECK-LABEL: vec_sro_i2u1
subroutine vec_sro_i2u1(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_i2u1

! CHECK-LABEL: vec_sro_i4u1
subroutine vec_sro_i4u1(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsro(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_i4u1

! CHECK-LABEL: vec_sro_u1u1
subroutine vec_sro_u1u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_u1u1

! CHECK-LABEL: vec_sro_u2u1
subroutine vec_sro_u2u1(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_u2u1

! CHECK-LABEL: vec_sro_u4u1
subroutine vec_sro_u4u1(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsro(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_u4u1

! CHECK-LABEL: vec_sro_r4u1
subroutine vec_sro_r4u1(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(1)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_sro_r4u1

!-------------------------------------

! CHECK-LABEL: vec_sro_i1u2
subroutine vec_sro_i1u2(arg1, arg2)
  vector(integer(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_i1u2

! CHECK-LABEL: vec_sro_i2u2
subroutine vec_sro_i2u2(arg1, arg2)
  vector(integer(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>
end subroutine vec_sro_i2u2

! CHECK-LABEL: vec_sro_i4u2
subroutine vec_sro_i4u2(arg1, arg2)
  vector(integer(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsro(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_i4u2

! CHECK-LABEL: vec_sro_u1u2
subroutine vec_sro_u1u2(arg1, arg2)
  vector(unsigned(1)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <16 x i8>
end subroutine vec_sro_u1u2

! CHECK-LABEL: vec_sro_u2u2
subroutine vec_sro_u2u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <8 x i16>

end subroutine vec_sro_u2u2

! CHECK-LABEL: vec_sro_u4u2
subroutine vec_sro_u4u2(arg1, arg2)
  vector(unsigned(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vsro(%[[arg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[arg1]], <4 x i32> %[[varg2]])
end subroutine vec_sro_u4u2

! CHECK-LABEL: vec_sro_r4u2
subroutine vec_sro_r4u2(arg1, arg2)
  vector(real(4)) :: arg1, r
  vector(unsigned(2)) :: arg2
  r = vec_sro(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[res:.*]] = fir.call @llvm.ppc.altivec.vsro(%[[bc1]], %[[bc2]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[vres:.*]] = fir.convert %[[res]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcres:.*]] = vector.bitcast %[[vres]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[bcres]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load {{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[varg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[varg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[res:.*]] = llvm.call @llvm.ppc.altivec.vsro(%[[varg1]], %[[varg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[res]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[varg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[varg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[res:.*]] = call <4 x i32> @llvm.ppc.altivec.vsro(<4 x i32> %[[varg1]], <4 x i32> %[[varg2]])
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[res]] to <4 x float>
end subroutine vec_sro_r4u2

!----------------------
! vec_sld
!----------------------

! CHECK-LABEL: vec_sld_test_i1i1
subroutine vec_sld_test_i1i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i1

! CHECK-LABEL: vec_sld_test_i1i2
subroutine vec_sld_test_i1i2(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i2

! CHECK-LABEL: vec_sld_test_i1i4
subroutine vec_sld_test_i1i4(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i4

! CHECK-LABEL: vec_sld_test_i1i8
subroutine vec_sld_test_i1i8(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i1i8

! CHECK-LABEL: vec_sld_test_i2i1
subroutine vec_sld_test_i2i1(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i1

! CHECK-LABEL: vec_sld_test_i2i2
subroutine vec_sld_test_i2i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i2

! CHECK-LABEL: vec_sld_test_i2i4
subroutine vec_sld_test_i2i4(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i4

! CHECK-LABEL: vec_sld_test_i2i8
subroutine vec_sld_test_i2i8(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i8

! CHECK-LABEL: vec_sld_test_i4i1
subroutine vec_sld_test_i4i1(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i1

! CHECK-LABEL: vec_sld_test_i4i2
subroutine vec_sld_test_i4i2(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i2

! CHECK-LABEL: vec_sld_test_i4i4
subroutine vec_sld_test_i4i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i4

! CHECK-LABEL: vec_sld_test_i4i8
subroutine vec_sld_test_i4i8(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i4i8

! CHECK-LABEL: vec_sld_test_u1i1
subroutine vec_sld_test_u1i1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i1

! CHECK-LABEL: vec_sld_test_u1i2
subroutine vec_sld_test_u1i2(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i2

! CHECK-LABEL: vec_sld_test_u1i4
subroutine vec_sld_test_u1i4(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i4

! CHECK-LABEL: vec_sld_test_u1i8
subroutine vec_sld_test_u1i8(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u1i8

! CHECK-LABEL: vec_sld_test_u2i1
subroutine vec_sld_test_u2i1(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i1

! CHECK-LABEL: vec_sld_test_u2i2
subroutine vec_sld_test_u2i2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i2

! CHECK-LABEL: vec_sld_test_u2i4
subroutine vec_sld_test_u2i4(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i4

! CHECK-LABEL: vec_sld_test_u2i8
subroutine vec_sld_test_u2i8(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u2i8

! CHECK-LABEL: vec_sld_test_u4i1
subroutine vec_sld_test_u4i1(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i1

! CHECK-LABEL: vec_sld_test_u4i2
subroutine vec_sld_test_u4i2(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i2

! CHECK-LABEL: vec_sld_test_u4i4
subroutine vec_sld_test_u4i4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i4

! CHECK-LABEL: vec_sld_test_u4i8
subroutine vec_sld_test_u4i8(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_u4i8

! CHECK-LABEL: vec_sld_test_r4i1
subroutine vec_sld_test_r4i1(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i1

! CHECK-LABEL: vec_sld_test_r4i2
subroutine vec_sld_test_r4i2(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i2

! CHECK-LABEL: vec_sld_test_r4i4
subroutine vec_sld_test_r4i4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i4

! CHECK-LABEL: vec_sld_test_r4i8
subroutine vec_sld_test_r4i8(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] : vector<16xi8>
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i8

!----------------------
! vec_sldw
!----------------------
! CHECK-LABEL: vec_sldw_test_i1i1
subroutine vec_sldw_test_i1i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i1

! CHECK-LABEL: vec_sldw_test_i1i2
subroutine vec_sldw_test_i1i2(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i2

! CHECK-LABEL: vec_sldw_test_i1i4
subroutine vec_sldw_test_i1i4(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i4

! CHECK-LABEL: vec_sldw_test_i1i8
subroutine vec_sldw_test_i1i8(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i1i8

! CHECK-LABEL: vec_sldw_test_i2i1
subroutine vec_sldw_test_i2i1(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i1

! CHECK-LABEL: vec_sldw_test_i2i2
subroutine vec_sldw_test_i2i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i2

! CHECK-LABEL: vec_sldw_test_i2i4
subroutine vec_sldw_test_i2i4(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i4

! CHECK-LABEL: vec_sldw_test_i2i8
subroutine vec_sldw_test_i2i8(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i2i8

! CHECK-LABEL: vec_sldw_test_i4i1
subroutine vec_sldw_test_i4i1(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i1

! CHECK-LABEL: vec_sldw_test_i4i2
subroutine vec_sldw_test_i4i2(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i2

! CHECK-LABEL: vec_sldw_test_i4i4
subroutine vec_sldw_test_i4i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i4

! CHECK-LABEL: vec_sldw_test_i4i8
subroutine vec_sldw_test_i4i8(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i4i8

! CHECK-LABEL: vec_sldw_test_i8i1
subroutine vec_sldw_test_i8i1(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i1

! CHECK-LABEL: vec_sldw_test_i8i2
subroutine vec_sldw_test_i8i2(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i2

! CHECK-LABEL: vec_sldw_test_i8i4
subroutine vec_sldw_test_i8i4(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i4

! CHECK-LABEL: vec_sldw_test_i8i8
subroutine vec_sldw_test_i8i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_i8i8

! CHECK-LABEL: vec_sldw_test_u1i1
subroutine vec_sldw_test_u1i1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i1

! CHECK-LABEL: vec_sldw_test_u1i2
subroutine vec_sldw_test_u1i2(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i2

! CHECK-LABEL: vec_sldw_test_u1i4
subroutine vec_sldw_test_u1i4(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i4

! CHECK-LABEL: vec_sldw_test_u1i8
subroutine vec_sldw_test_u1i8(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg2]], %[[carg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg2]], %[[arg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u1i8

! CHECK-LABEL: vec_sldw_test_u2i1
subroutine vec_sldw_test_u2i1(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i1

! CHECK-LABEL: vec_sldw_test_u2i2
subroutine vec_sldw_test_u2i2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i2

! CHECK-LABEL: vec_sldw_test_u2i4
subroutine vec_sldw_test_u2i4(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i4

! CHECK-LABEL: vec_sldw_test_u2i8
subroutine vec_sldw_test_u2i8(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u2i8

! CHECK-LABEL: vec_sldw_test_u4i1
subroutine vec_sldw_test_u4i1(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i1

! CHECK-LABEL: vec_sldw_test_u4i2
subroutine vec_sldw_test_u4i2(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i2

! CHECK-LABEL: vec_sldw_test_u4i4
subroutine vec_sldw_test_u4i4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i4

! CHECK-LABEL: vec_sldw_test_u4i8
subroutine vec_sldw_test_u4i8(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u4i8

! CHECK-LABEL: vec_sldw_test_u8i1
subroutine vec_sldw_test_u8i1(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i1

! CHECK-LABEL: vec_sldw_test_u8i2
subroutine vec_sldw_test_u8i2(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i2

! CHECK-LABEL: vec_sldw_test_u8i4
subroutine vec_sldw_test_u8i4(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i4

! CHECK-LABEL: vec_sldw_test_u8i8
subroutine vec_sldw_test_u8i8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_u8i8

! CHECK-LABEL: vec_sldw_test_r4i1
subroutine vec_sldw_test_r4i1(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i1

! CHECK-LABEL: vec_sldw_test_r4i2
subroutine vec_sldw_test_r4i2(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i2

! CHECK-LABEL: vec_sldw_test_r4i4
subroutine vec_sldw_test_r4i4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i4

! CHECK-LABEL: vec_sldw_test_r4i8
subroutine vec_sldw_test_r4i8(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r4i8

! CHECK-LABEL: vec_sldw_test_r8i1
subroutine vec_sldw_test_r8i1(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i1

! CHECK-LABEL: vec_sldw_test_r8i2
subroutine vec_sldw_test_r8i2(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i2

! CHECK-LABEL: vec_sldw_test_r8i4
subroutine vec_sldw_test_r8i4(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i4

! CHECK-LABEL: vec_sldw_test_r8i8
subroutine vec_sldw_test_r8i8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_sldw(arg1, arg2, 3_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<16xi8>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[barg2]], %[[barg1]] [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] : vector<16xi8> 
! CHECK-LLVMIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[br]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i8
