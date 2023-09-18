! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="MLIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------
! vec_splat
!----------------

! CHECK-LABEL: vec_splat_testi8i8
subroutine vec_splat_testi8i8(x)
  vector(integer(1)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi8i8

! CHECK-LABEL: vec_splat_testi8i16
subroutine vec_splat_testi8i16(x)
  vector(integer(1)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi8i16

! CHECK-LABEL: vec_splat_testi8i32
subroutine vec_splat_testi8i32(x)
  vector(integer(1)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi8i32

! CHECK-LABEL: vec_splat_testi8i64
subroutine vec_splat_testi8i64(x)
  vector(integer(1)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi8i64

! CHECK-LABEL: vec_splat_testi16i8
subroutine vec_splat_testi16i8(x)
  vector(integer(2)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi16i8

! CHECK-LABEL: vec_splat_testi16i16
subroutine vec_splat_testi16i16(x)
  vector(integer(2)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi16i16

! CHECK-LABEL: vec_splat_testi16i32
subroutine vec_splat_testi16i32(x)
  vector(integer(2)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi16i32

! CHECK-LABEL: vec_splat_testi16i64
subroutine vec_splat_testi16i64(x)
  vector(integer(2)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi16i64

! CHECK-LABEL: vec_splat_testi32i8
subroutine vec_splat_testi32i8(x)
  vector(integer(4)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi32i8

! CHECK-LABEL: vec_splat_testi32i16
subroutine vec_splat_testi32i16(x)
  vector(integer(4)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi32i16

! CHECK-LABEL: vec_splat_testi32i32
subroutine vec_splat_testi32i32(x)
  vector(integer(4)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi32i32

! CHECK-LABEL: vec_splat_testi32i64
subroutine vec_splat_testi32i64(x)
  vector(integer(4)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi32i64

! CHECK-LABEL: vec_splat_testi64i8
subroutine vec_splat_testi64i8(x)
  vector(integer(8)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi64i8

! CHECK-LABEL: vec_splat_testi64i16
subroutine vec_splat_testi64i16(x)
  vector(integer(8)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi64i16

! CHECK-LABEL: vec_splat_testi64i32
subroutine vec_splat_testi64i32(x)
  vector(integer(8)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi64i32

! CHECK-LABEL: vec_splat_testi64i64
subroutine vec_splat_testi64i64(x)
  vector(integer(8)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testi64i64

! CHECK-LABEL: vec_splat_testf32i8
subroutine vec_splat_testf32i8(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<4xf32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<4xf32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xf32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xf32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x float> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf32i8

! CHECK-LABEL: vec_splat_testf32i16
subroutine vec_splat_testf32i16(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<4xf32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<4xf32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xf32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xf32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x float> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf32i16

! CHECK-LABEL: vec_splat_testf32i32
subroutine vec_splat_testf32i32(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<4xf32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<4xf32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xf32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xf32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x float> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf32i32

! CHECK-LABEL: vec_splat_testf32i64
subroutine vec_splat_testf32i64(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<4xf32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<4xf32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xf32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xf32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x float> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf32i64

! CHECK-LABEL: vec_splat_testf64i8
subroutine vec_splat_testf64i8(x)
  vector(real(8)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! FIR: %[[c:.*]] = arith.constant 2 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<2xf64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xf64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<2xf64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xf64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xf64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xf64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x double> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x double> undef, double %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x double> %[[ins]], <2 x double> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x double> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf64i8

! CHECK-LABEL: vec_splat_testf64i16
subroutine vec_splat_testf64i16(x)
  vector(real(8)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! FIR: %[[c:.*]] = arith.constant 2 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<2xf64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xf64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<2xf64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xf64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xf64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xf64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x double> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x double> undef, double %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x double> %[[ins]], <2 x double> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x double> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf64i16

! CHECK-LABEL: vec_splat_testf64i32
subroutine vec_splat_testf64i32(x)
  vector(real(8)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! FIR: %[[c:.*]] = arith.constant 2 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<2xf64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xf64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<2xf64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xf64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xf64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xf64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x double> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x double> undef, double %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x double> %[[ins]], <2 x double> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x double> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf64i32

! CHECK-LABEL: vec_splat_testf64i64
subroutine vec_splat_testf64i64(x)
  vector(real(8)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! FIR: %[[c:.*]] = arith.constant 2 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<2xf64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xf64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<2xf64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xf64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xf64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xf64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x double> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x double> undef, double %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x double> %[[ins]], <2 x double> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x double> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf64i64

! CHECK-LABEL: vec_splat_testu8i8
subroutine vec_splat_testu8i8(x)
  vector(unsigned(1)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i8

! CHECK-LABEL: vec_splat_testu8i16
subroutine vec_splat_testu8i16(x)
  vector(unsigned(1)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i16

! CHECK-LABEL: vec_splat_testu8i32
subroutine vec_splat_testu8i32(x)
  vector(unsigned(1)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i32

! CHECK-LABEL: vec_splat_testu8i64
subroutine vec_splat_testu8i64(x)
  vector(unsigned(1)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(16 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<16xi8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i64

! CHECK-LABEL: vec_splat_testu16i8
subroutine vec_splat_testu16i8(x)
  vector(unsigned(2)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu16i8

! CHECK-LABEL: vec_splat_testu16i16
subroutine vec_splat_testu16i16(x)
  vector(unsigned(2)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu16i16

! CHECK-LABEL: vec_splat_testu16i32
subroutine vec_splat_testu16i32(x)
  vector(unsigned(2)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu16i32

! CHECK-LABEL: vec_splat_testu16i64
subroutine vec_splat_testu16i64(x)
  vector(unsigned(2)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! FIR: %[[c:.*]] = arith.constant 8 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<8xi16>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(8 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<8xi16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <8 x i16> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu16i64

! CHECK-LABEL: vec_splat_testu32i8
subroutine vec_splat_testu32i8(x)
  vector(unsigned(4)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu32i8

! CHECK-LABEL: vec_splat_testu32i16
subroutine vec_splat_testu32i16(x)
  vector(unsigned(4)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu32i16

! CHECK-LABEL: vec_splat_testu32i32
subroutine vec_splat_testu32i32(x)
  vector(unsigned(4)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu32i32

! CHECK-LABEL: vec_splat_testu32i64
subroutine vec_splat_testu32i64(x)
  vector(unsigned(4)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! FIR: %[[c:.*]] = arith.constant 4 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<4xi32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(4 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<4xi32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x i32> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu32i64

! CHECK-LABEL: vec_splat_testu64i8
subroutine vec_splat_testu64i8(x)
  vector(unsigned(8)) :: x, y
  y = vec_splat(x, 0_1)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i8
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i8
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i8] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i8) : i8
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i8) : i8
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i8
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i8] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i8 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu64i8

! CHECK-LABEL: vec_splat_testu64i16
subroutine vec_splat_testu64i16(x)
  vector(unsigned(8)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i16] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i16) : i16
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i16) : i16
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i16] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i16 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu64i16

! CHECK-LABEL: vec_splat_testu64i32
subroutine vec_splat_testu64i32(x)
  vector(unsigned(8)) :: x, y
  y = vec_splat(x, 0_4)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i32
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i32
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i32] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i32) : i32
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i32
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i32] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i32 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu64i32

! CHECK-LABEL: vec_splat_testu64i64
subroutine vec_splat_testu64i64(x)
  vector(unsigned(8)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! FIR: %[[c:.*]] = arith.constant 2 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[u]] : i64] : vector<2xi64>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! MLIR: %[[idx:.*]] = llvm.mlir.constant(0 : i64) : i64
! MLIR: %[[c:.*]] = llvm.mlir.constant(2 : i64) : i64
! MLIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! MLIR: %[[ele:.*]] = llvm.extractelement %[[x]][%[[u]] : i64] : vector<2xi64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[ele]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <2 x i64> %[[x]], i64 0
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu64i64

!----------------
! vec_splats
!----------------

! CHECK-LABEL: vec_splats_testi8
subroutine vec_splats_testi8(x)
  integer(1) :: x
  vector(integer(1)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<i8>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<16xi8>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<16xi8>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi8>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! LLVMIR: %[[x:.*]] = load i8, ptr %{{[0-9]}}, align 1
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testi8

! CHECK-LABEL: vec_splats_testi16
subroutine vec_splats_testi16(x)
  integer(2) :: x
  vector(integer(2)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<8xi16>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<i16>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<8xi16>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xi16>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! LLVMIR: %[[x:.*]] = load i16, ptr %{{[0-9]}}, align 2
! LLVMIR: %[[ins:.*]] = insertelement <8 x i16> undef, i16 %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <8 x i16> %[[ins]], <8 x i16> undef, <8 x i32> zeroinitializer
! LLVMIR: store <8 x i16> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testi16

! CHECK-LABEL: vec_splats_testi32
subroutine vec_splats_testi32(x)
  integer(4) :: x
  vector(integer(4)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<i32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: %[[x:.*]] = load i32, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[ins:.*]] = insertelement <4 x i32> undef, i32 %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x i32> %[[ins]], <4 x i32> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x i32> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testi32

! CHECK-LABEL: vec_splats_testi64
subroutine vec_splats_testi64(x)
  integer(8) :: x
  vector(integer(8)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<2xi64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<i64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xi64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<2xi64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xi64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! LLVMIR: %[[x:.*]] = load i64, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[ins:.*]] = insertelement <2 x i64> undef, i64 %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x i64> %[[ins]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testi64

! CHECK-LABEL: vec_splats_testf32
subroutine vec_splats_testf32(x)
  real(4) :: x
  vector(real(4)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<f32>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<f32>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<4xf32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0, 0, 0] : vector<4xf32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! LLVMIR: %[[x:.*]] = load float, ptr %{{[0-9]}}, align 4
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testf32

! CHECK-LABEL: vec_splats_testf64
subroutine vec_splats_testf64(x)
  real(8) :: x
  vector(real(8)) :: y
  y = vec_splats(x)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<f64>
! FIR: %[[vy:.*]] = vector.splat %[[x]] : vector<2xf64>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! MLIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<f64>
! MLIR: %[[undef:.*]] = llvm.mlir.undef : vector<2xf64>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[x]], %[[undef]][%[[zero]] : i32] : vector<2xf64>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[undef]] [0, 0] : vector<2xf64>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! LLVMIR: %[[x:.*]] = load double, ptr %{{[0-9]}}, align 8
! LLVMIR: %[[ins:.*]] = insertelement <2 x double> undef, double %[[x]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <2 x double> %[[ins]], <2 x double> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x double> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splats_testf64

! CHECK-LABEL: vec_splat_s32testi8
subroutine vec_splat_s32testi8()
  vector(integer(4)) :: y
  y = vec_splat_s32(7_1)
! FIR: %[[val:.*]] = arith.constant 7 : i8
! FIR: %[[cval:.*]] = fir.convert %[[val]] : (i8) -> i32
! FIR: %[[vy:.*]] = vector.splat %[[cval]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[val:.*]] = llvm.mlir.constant(7 : i8) : i8
! MLIR: %[[cval:.*]] = llvm.sext %[[val]] : i8 to i32
! MLIR: %[[und:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[cval]], %[[und]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[und]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: store <4 x i32> <i32 7, i32 7, i32 7, i32 7>, ptr %{{[0-9]}}, align 16
end subroutine vec_splat_s32testi8

! CHECK-LABEL: vec_splat_s32testi16
subroutine vec_splat_s32testi16()
  vector(integer(4)) :: y
  y = vec_splat_s32(7_2)
! FIR: %[[val:.*]] = arith.constant 7 : i16
! FIR: %[[cval:.*]] = fir.convert %[[val]] : (i16) -> i32
! FIR: %[[vy:.*]] = vector.splat %[[cval]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[val:.*]] = llvm.mlir.constant(7 : i16) : i16
! MLIR: %[[cval:.*]] = llvm.sext %[[val]] : i16 to i32
! MLIR: %[[und:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[cval]], %[[und]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[und]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: store <4 x i32> <i32 7, i32 7, i32 7, i32 7>, ptr %{{[0-9]}}, align 16
end subroutine vec_splat_s32testi16

! CHECK-LABEL: vec_splat_s32testi32
subroutine vec_splat_s32testi32()
  vector(integer(4)) :: y
  y = vec_splat_s32(7_4)
! FIR: %[[val:.*]] = arith.constant 7 : i32
! FIR: %[[vy:.*]] = vector.splat %[[val]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[val:.*]] = llvm.mlir.constant(7 : i32) : i32
! MLIR: %[[y:.*]] = llvm.mlir.constant(dense<7> : vector<4xi32>) : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: store <4 x i32> <i32 7, i32 7, i32 7, i32 7>, ptr %{{[0-9]}}, align 16
end subroutine vec_splat_s32testi32

! CHECK-LABEL: vec_splat_s32testi64
subroutine vec_splat_s32testi64()
  vector(integer(4)) :: y
  y = vec_splat_s32(7_8)
! FIR: %[[val:.*]] = arith.constant 7 : i64
! FIR: %[[cval:.*]] = fir.convert %[[val]] : (i64) -> i32
! FIR: %[[vy:.*]] = vector.splat %[[cval]] : vector<4xi32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! MLIR: %[[val:.*]] = llvm.mlir.constant(7 : i64) : i64
! MLIR: %[[cval:.*]] = llvm.trunc %[[val]] : i64 to i32
! MLIR: %[[und:.*]] = llvm.mlir.undef : vector<4xi32>
! MLIR: %[[zero:.*]] = llvm.mlir.constant(0 : i32) : i32
! MLIR: %[[ins:.*]] = llvm.insertelement %[[cval]], %[[und]][%[[zero]] : i32] : vector<4xi32>
! MLIR: %[[y:.*]] = llvm.shufflevector %[[ins]], %[[und]] [0, 0, 0, 0] : vector<4xi32>
! MLIR: llvm.store %[[y]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! LLVMIR: store <4 x i32> <i32 7, i32 7, i32 7, i32 7>, ptr %{{[0-9]}}, align 16
end subroutine vec_splat_s32testi64
