! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_perm_test_i1
subroutine vec_perm_test_i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! CHECK: store <16 x i8> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i1

! CHECK-LABEL: vec_perm_test_i2
subroutine vec_perm_test_i2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! CHECK: store <8 x i16> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i2

! CHECK-LABEL: vec_perm_test_i4
subroutine vec_perm_test_i4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[carg2]], %[[carg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[call]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[arg2]], %[[arg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[call]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]], <16 x i8> %[[xor]])
! CHECK: store <4 x i32> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i4

! CHECK-LABEL: vec_perm_test_i8
subroutine vec_perm_test_i8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x i64>
! CHECK: store <2 x i64> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_i8

! CHECK-LABEL: vec_perm_test_u1
subroutine vec_perm_test_u1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<16xi8>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <16 x i8> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! CHECK: store <16 x i8> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u1

! CHECK-LABEL: vec_perm_test_u2
subroutine vec_perm_test_u2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<8xi16>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! CHECK: store <8 x i16> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u2

! CHECK-LABEL: vec_perm_test_u4
subroutine vec_perm_test_u4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[carg2]], %[[carg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[call2]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[arg2]], %[[arg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[call]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[arg2]], <4 x i32> %[[arg1]], <16 x i8> %[[xor]])
! CHECK: store <4 x i32> %[[call]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u4

! CHECK-LABEL: vec_perm_test_u8
subroutine vec_perm_test_u8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xi64> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x i64>
! CHECK: store <2 x i64> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_u8

! CHECK-LABEL: vec_perm_test_r4
subroutine vec_perm_test_r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! CHECK: store <4 x float> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_r4

! CHECK-LABEL: vec_perm_test_r8
subroutine vec_perm_test_r8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_perm(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg3:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg3:.*]] = fir.convert %[[arg3]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<4xi32>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<4xi32>
! CHECK-FIR: %[[const:.*]] = arith.constant -1 : i8
! CHECK-FIR: %[[vconst:.*]] = vector.broadcast %[[const]] : i8 to vector<16xi8>
! CHECK-FIR: %[[xor:.*]] = arith.xori %[[carg3]], %[[vconst]] : vector<16xi8>
! CHECK-FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) fastmath<contract> : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> !fir.vector<4:i32>
! CHECK-FIR: %[[call2:.*]] = fir.convert %[[call]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcall:.*]] = llvm.bitcast %[[call2]] : vector<4xi32> to vector<2xf64>
! CHECK-FIR: %[[ccall:.*]] = fir.convert %[[bcall]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[ccall]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<4xi32>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<4xi32>
! CHECK-LLVMIR: %[[const:.*]] = llvm.mlir.constant(-1 : i8) : i8
! CHECK-LLVMIR: %[[vconst:.*]] = llvm.mlir.constant(dense<-1> : vector<16xi8>) : vector<16xi8>
! CHECK-LLVMIR: %[[xor:.*]] = llvm.xor %[[arg3]], %[[vconst]]  : vector<16xi8>
! CHECK-LLVMIR: %[[call:.*]] = llvm.call @llvm.ppc.altivec.vperm(%[[barg2]], %[[barg1]], %[[xor]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>, vector<16xi8>) -> vector<4xi32>
! CHECK-LLVMIR: %[[bcall:.*]] = llvm.bitcast %[[call]] : vector<4xi32> to vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[bcall]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <4 x i32>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <4 x i32>
! CHECK: %[[xor:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! CHECK: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> %[[barg2]], <4 x i32> %[[barg1]], <16 x i8> %[[xor]])
! CHECK: %[[bcall:.*]] = bitcast <4 x i32> %[[call]] to <2 x double>
! CHECK: store <2 x double> %[[bcall]], ptr %{{.*}}, align 16
end subroutine vec_perm_test_r8

! CHECK-LABEL: vec_permi_test_i8i1
subroutine vec_permi_test_i8i1(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i1

! CHECK-LABEL: vec_permi_test_i8i2
subroutine vec_permi_test_i8i2(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 2>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i2

! CHECK-LABEL: vec_permi_test_i8i4
subroutine vec_permi_test_i8i4(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 3>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i4

! CHECK-LABEL: vec_permi_test_i8i8
subroutine vec_permi_test_i8i8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_i8i8

! CHECK-LABEL: vec_permi_test_u8i1
subroutine vec_permi_test_u8i1(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i1

! CHECK-LABEL: vec_permi_test_u8i2
subroutine vec_permi_test_u8i2(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 2>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i2

! CHECK-LABEL: vec_permi_test_u8i4
subroutine vec_permi_test_u8i4(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 3>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i4

! CHECK-LABEL: vec_permi_test_u8i8
subroutine vec_permi_test_u8i8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x i64> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_u8i8

! CHECK-LABEL: vec_permi_test_r4i1
subroutine vec_permi_test_r4i1(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [1, 3] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[bshuf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[barg1]], %[[barg2]] [1, 3] : vector<2xf64>
! CHECK-LLVMIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[bshuf]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 1, i32 3>
! CHECK: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! CHECK: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i1

! CHECK-LABEL: vec_permi_test_r4i2
subroutine vec_permi_test_r4i2(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [1, 2] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[bshuf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[barg1]], %[[barg2]] [1, 2] : vector<2xf64>
! CHECK-LLVMIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[bshuf]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 1, i32 2>
! CHECK: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! CHECK: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i2

! CHECK-LABEL: vec_permi_test_r4i4
subroutine vec_permi_test_r4i4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [0, 3] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[bshuf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[barg1]], %[[barg2]] [0, 3] : vector<2xf64>
! CHECK-LLVMIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[bshuf]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 0, i32 3>
! CHECK: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! CHECK: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i4

! CHECK-LABEL: vec_permi_test_r4i8
subroutine vec_permi_test_r4i8(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [0, 2] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[bshuf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[barg1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[barg2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<2xf64>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[barg1]], %[[barg2]] [0, 2] : vector<2xf64>
! CHECK-LLVMIR: %[[bshuf:.*]] = llvm.bitcast %[[shuf]] : vector<2xf64> to vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[bshuf]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <2 x double>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <2 x double>
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[barg1]], <2 x double> %[[barg2]], <2 x i32> <i32 0, i32 2>
! CHECK: %[[bshuf:.*]] = bitcast <2 x double> %[[shuf]] to <4 x float>
! CHECK: store <4 x float> %[[bshuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r4i8

! CHECK-LABEL: vec_permi_test_r8i1
subroutine vec_permi_test_r8i1(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 3_1)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [1, 3] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i1

! CHECK-LABEL: vec_permi_test_r8i2
subroutine vec_permi_test_r8i2(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 2_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [1, 2] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 2] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 2>
! CHECK: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i2

! CHECK-LABEL: vec_permi_test_r8i4
subroutine vec_permi_test_r8i4(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 1_4)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [0, 3] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 3] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 3>
! CHECK: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i4

! CHECK-LABEL: vec_permi_test_r8i8
subroutine vec_permi_test_r8i8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  r = vec_permi(arg1, arg2, 0_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<2xf64>
! CHECK-FIR: %[[shuf:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [0, 2] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cshuf:.*]] = fir.convert %[[shuf]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cshuf]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[shuf:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[shuf]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[shuf:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x double> %[[shuf]], ptr %{{.*}}, align 16
end subroutine vec_permi_test_r8i8
