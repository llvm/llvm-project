! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_st
!----------------------

! CHECK-LABEL: vec_st_vi1i2vi1
subroutine vec_st_vi1i2vi1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<16:i8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[bcArg1:.*]] = vector.bitcast %[[cnvArg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<16xi8>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[bcArg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: %[[bcArg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vi1i2vi1

! CHECK-LABEL: vec_st_vi2i2vi2
subroutine vec_st_vi2i2vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[bcArg1:.*]] = vector.bitcast %[[cnvArg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[bcArg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: %[[bcArg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vi2i2vi2

! CHECK-LABEL: vec_st_vi4i2vi4
subroutine vec_st_vi4i2vi4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg3
  integer(2) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[varg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_st_vi4i2vi4

! CHECK-LABEL: vec_st_vu1i4vu1
subroutine vec_st_vu1i4vu1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<16:ui8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[bcArg1:.*]] = vector.bitcast %[[cnvArg1]] : vector<16xi8> to vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<16xi8>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[bcArg1:.*]] = llvm.bitcast %[[arg1]] : vector<16xi8> to vector<4xi32>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: %[[bcArg1:.*]] = bitcast <16 x i8> %[[arg1]] to <4 x i32>
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vu1i4vu1

! CHECK-LABEL: vec_st_vu2i4vu2
subroutine vec_st_vu2i4vu2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<8:ui16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[bcArg1:.*]] = vector.bitcast %[[cnvArg1]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[bcArg1:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[bcArg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: %[[bcArg1:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[bcArg1]], ptr %[[arg3]])
end subroutine vec_st_vu2i4vu2

! CHECK-LABEL: vec_st_vu4i4vu4
subroutine vec_st_vu4i4vu4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg3
  integer(4) :: arg2
  call vec_st(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:ui32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[varg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: call void @llvm.ppc.altivec.stvx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_st_vu4i4vu4

! CHECK-LABEL: vec_st_vi4i4via4
subroutine vec_st_vi4i4via4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1, arg3(5)
  integer(4) :: arg2, i
  call vec_st(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[cnst:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[cnstm1:.*]] = arith.subi %[[idx64]], %[[cnst]] : i64
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %arg2, %[[cnstm1]] : (!fir.ref<!fir.array<5x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[ref:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[pos:.*]] = fir.coordinate_of %[[ref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvx(%[[varg1]], %[[pos]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]] : i64
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<5 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[bc:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[pos:.*]] = llvm.getelementptr %[[bc]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvx(%[[arg1]], %[[pos]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK:  %5 = load <4 x i32>, ptr %0, align 16
! CHECK:  %6 = load i32, ptr %1, align 4
! CHECK:  %7 = load i32, ptr %3, align 4
! CHECK:  %8 = sext i32 %7 to i64
! CHECK:  %9 = sub i64 %8, 1
! CHECK:  %10 = getelementptr [5 x <4 x i32>], ptr %2, i32 0, i64 %9
! CHECK:  %11 = getelementptr i8, ptr %10, i32 %6
! CHECK:  call void @llvm.ppc.altivec.stvx(<4 x i32> %5, ptr %11)
end subroutine vec_st_vi4i4via4

!----------------------
! vec_ste
!----------------------

! CHECK-LABEL: vec_ste_vi1i2i1
subroutine vec_ste_vi1i2i1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1
  integer(2) :: arg2
  integer(1) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvebx(%[[cnvArg1]], %[[addr]]) fastmath<contract> : (vector<16xi8>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3:.*]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvebx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: call void @llvm.ppc.altivec.stvebx(<16 x i8> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi1i2i1

! CHECK-LABEL: vec_ste_vi2i2i2
subroutine vec_ste_vi2i2i2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(2) :: arg2
  integer(2) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i16>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvehx(%[[cnvArg1]], %[[addr]]) fastmath<contract> : (vector<8xi16>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<i16> to !llvm.ptr<i8> 
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvehx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: call void @llvm.ppc.altivec.stvehx(<8 x i16> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi2i2i2

! CHECK-LABEL: vec_ste_vi4i2i4
subroutine vec_ste_vi4i2i4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1
  integer(2) :: arg2
  integer(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvewx(%[[varg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvewx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i16 %5
! CHECK: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vi4i2i4

! CHECK-LABEL: vec_ste_vu1i4u1
subroutine vec_ste_vu1i4u1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1
  integer(4) :: arg2
  integer(1) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvebx(%[[cnvArg1]], %[[addr]]) fastmath<contract> : (vector<16xi8>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3:.*]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvebx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: call void @llvm.ppc.altivec.stvebx(<16 x i8> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu1i4u1

! CHECK-LABEL: vec_ste_vu2i4u2
subroutine vec_ste_vu2i4u2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1
  integer(4) :: arg2
  integer(2) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i16>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvehx(%[[cnvArg1]], %[[addr]]) fastmath<contract> : (vector<8xi16>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i16> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvehx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: call void @llvm.ppc.altivec.stvehx(<8 x i16> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu2i4u2

! CHECK-LABEL: vec_ste_vu4i4u4
subroutine vec_ste_vu4i4u4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1
  integer(4) :: arg2
  integer(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32> 
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvewx(%[[varg1]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvewx(%[[arg1]], %[[addr]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %{{.*}}, align 4
! CHECK: %[[arg3:.*]] = getelementptr i8, ptr %{{.*}}, i32 %5
! CHECK: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[arg3]])
end subroutine vec_ste_vu4i4u4

! CHECK-LABEL: vec_ste_vr4i4r4
subroutine vec_ste_vr4i4r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(4) :: arg2
  real(4) :: arg3
  call vec_ste(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[pos:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[cnvArg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bc:.*]] = vector.bitcast %[[cnvArg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvewx(%[[bc]], %[[pos]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[pos:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[bc:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvewx(%[[bc]], %[[pos]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[pos:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK: %[[bc:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[bc]], ptr %[[pos]])

end subroutine vec_ste_vr4i4r4

! CHECK-LABEL: vec_ste_vi4i4ia4
subroutine vec_ste_vi4i4ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2, i
  integer(4) :: arg3(5)
  call vec_ste(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[cnst:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[cnstm1:.*]] = arith.subi %[[idx64]], %[[cnst]] : i64
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %arg2, %[[cnstm1]] : (!fir.ref<!fir.array<5xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[ref:.*]] = fir.convert %[[addr]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[pos:.*]] = fir.coordinate_of %[[ref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: fir.call @llvm.ppc.altivec.stvewx(%[[varg1]], %[[pos]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<5 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[bc:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[pos:.*]] = llvm.getelementptr %[[bc]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: llvm.call @llvm.ppc.altivec.stvewx(%[[arg1]], %[[pos]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, !llvm.ptr<i8>) -> ()

! CHECK:  %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK:  %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK:  %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK:  %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK:  %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK:  %[[addr:.*]] = getelementptr [5 x i32], ptr %[[arg3:.*]], i32 0, i64 %[[idx64m1]]
! CHECK:  %[[pos:.*]] = getelementptr i8, ptr %[[addr]], i32 %[[arg2]]
! CHECK:  call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[arg1]], ptr %[[pos]])
end subroutine vec_ste_vi4i4ia4

!----------------------
! vec_stxv
!----------------------

! CHECK-LABEL: vec_stxv_test_vr4i2r4
subroutine vec_stxv_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_stxv(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3ptr:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3ptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3ptr]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! CHECK: store <4 x float> %[[arg1]], ptr %[[addr]], align 1
end subroutine vec_stxv_test_vr4i2r4

! CHECK-LABEL: vec_stxv_test_vi4i8ia4
subroutine vec_stxv_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_stxv(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %arg2, %[[idx64m1]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[elemref:.*]] = fir.convert %[[elem]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<10 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[elemref:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemref]][%[[arg2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i64, ptr %1, align 8
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [10 x i32], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i64 %6
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_stxv_test_vi4i8ia4

! CHECK-LABEL: vec_stxv_test_vi2i4vi2
subroutine vec_stxv_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_stxv(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK:  store <8 x i16> %[[arg1]], ptr %[[addr]], align 1
end subroutine vec_stxv_test_vi2i4vi2

! CHECK-LABEL: vec_stxv_test_vi4i4vai4
subroutine vec_stxv_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_stxv(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %[[arg3:.*]], %[[idx64m1]] : (!fir.ref<!fir.array<20x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[elemptr:.*]] = fir.convert %[[elem]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %[[arg3:.*]][0, %[[idx64m1]]] : (!llvm.ptr<array<20 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[elemptr:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemptr]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [20 x <4 x i32>], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i32 %[[arg2]]
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_stxv_test_vi4i4vai4

!----------------------
! vec_xst
!----------------------

! CHECK-LABEL: vec_xst_test_vr4i2r4
subroutine vec_xst_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xst(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3ptr:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3ptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3ptr]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]
  
! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! CHECK: store <4 x float> %[[arg1]], ptr %[[addr]], align 1
end subroutine vec_xst_test_vr4i2r4

! CHECK-LABEL: vec_xst_test_vi4i8ia4
subroutine vec_xst_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xst(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %arg2, %[[idx64m1]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[elemref:.*]] = fir.convert %[[elem]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<10 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[elemref:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemref]][%[[arg2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i64, ptr %1, align 8
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [10 x i32], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i64 %6
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_xst_test_vi4i8ia4

! CHECK-LABEL: vec_xst_test_vi2i4vi2
subroutine vec_xst_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xst(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK:  store <8 x i16> %[[arg1]], ptr %[[addr]], align 1
end subroutine vec_xst_test_vi2i4vi2

! CHECK-LABEL: vec_xst_test_vi4i4vai4
subroutine vec_xst_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xst(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %[[arg3:.*]], %[[idx64m1]] : (!fir.ref<!fir.array<20x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[elemptr:.*]] = fir.convert %[[elem]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: fir.store %[[arg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %[[arg3:.*]][0, %[[idx64m1]]] : (!llvm.ptr<array<20 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[elemptr:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemptr]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [20 x <4 x i32>], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i32 %[[arg2]]
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_xst_test_vi4i4vai4

!----------------------
! vec_xst_be
!----------------------

! CHECK-LABEL: vec_xst_be_test_vr4i2r4
subroutine vec_xst_be_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xst_be(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3ptr:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3ptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[undef:.*]] = fir.undefined vector<4xf32>
! CHECK-FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[undef]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! CHECK-FIR: %[[fvarg1:.*]] = fir.convert %[[shf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[fvarg1]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3ptr]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xf32>
! CHECK-LLVMIR: %[[shf:.*]] = llvm.shufflevector %[[arg1]], %[[undef]] [3, 2, 1, 0] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[shf]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! CHECK: %[[shf:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! CHECK: store <4 x float> %[[shf]], ptr %[[addr]], align 1
end subroutine vec_xst_be_test_vr4i2r4

! CHECK-LABEL: vec_xst_be_test_vi4i8ia4
subroutine vec_xst_be_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xst_be(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %arg2, %[[idx64m1]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[elemref:.*]] = fir.convert %[[elem]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[undef:.*]] = fir.undefined vector<4xi32>
! CHECK-FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[undef]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<10 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[elemref:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemref]][%[[arg2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! CHECK-LLVMIR: %[[src:.*]] = llvm.shufflevector %[[arg1]], %[[undef]] [3, 2, 1, 0] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i64, ptr %1, align 8
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [10 x i32], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i64 %6
! CHECK: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! CHECK: store <4 x i32> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xst_be_test_vi4i8ia4

! CHECK-LABEL: vec_xst_be_test_vi2i4vi2
subroutine vec_xst_be_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xst_be(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[undef:.*]] = fir.undefined vector<8xi16>
! CHECK-FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[undef]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
! CHECK-FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[undef:.*]] = llvm.mlir.undef : vector<8xi16>
! CHECK-LLVMIR: %[[src:.*]] = llvm.shufflevector %[[arg1]], %[[undef]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK: %[[src:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! CHECK:  store <8 x i16> %[[src]], ptr %[[addr]], align 1
end subroutine vec_xst_be_test_vi2i4vi2

! CHECK-LABEL: vec_xst_be_test_vi4i4vai4
subroutine vec_xst_be_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xst_be(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %[[arg3:.*]], %[[idx64m1]] : (!fir.ref<!fir.array<20x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[elemptr:.*]] = fir.convert %[[elem]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[undef:.*]] = fir.undefined vector<4xi32>
! CHECK-FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[undef]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %[[arg3:.*]][0, %[[idx64m1]]] : (!llvm.ptr<array<20 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[elemptr:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemptr]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[undef:.*]] = llvm.mlir.undef : vector<4xi32>
! CHECK-LLVMIR: %[[src:.*]] = llvm.shufflevector %[[arg1]], %[[undef]] [3, 2, 1, 0] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [20 x <4 x i32>], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i32 %[[arg2]]
! CHECK: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! CHECK: store <4 x i32> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xst_be_test_vi4i4vai4

!----------------------
! vec_xstd2
!----------------------

! CHECK-LABEL: vec_xstd2_test_vr4i2r4
subroutine vec_xstd2_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xstd2(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3ptr:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3ptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bcsrc:.*]] = vector.bitcast %[[vsrc]] : vector<4xf32> to vector<2xi64>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[bcsrc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3ptr]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[src:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<2xi64>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]
  
! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! CHECK: %[[src:.*]] = bitcast <4 x float> %[[arg1]] to <2 x i64>
! CHECK: store <2 x i64> %[[src]], ptr %[[addr]], align 1
end subroutine vec_xstd2_test_vr4i2r4

! CHECK-LABEL: vec_xstd2_test_vi4i8ia4
subroutine vec_xstd2_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %arg2, %[[idx64m1]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[elemref:.*]] = fir.convert %[[elem]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcsrc:.*]] = vector.bitcast %[[vsrc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[bcsrc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<10 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[elemref:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemref]][%[[arg2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[src:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i64, ptr %1, align 8
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [10 x i32], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i64 %6
! CHECK: %[[src:.*]] = bitcast <4 x i32> %[[arg1]] to <2 x i64>
! CHECK: store <2 x i64> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xstd2_test_vi4i8ia4

! CHECK-LABEL: vec_xstd2_test_vi2i4vi2
subroutine vec_xstd2_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xstd2(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[bcsrc:.*]] = vector.bitcast %[[vsrc]] : vector<8xi16> to vector<2xi64>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[bcsrc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[src:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<2xi64>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK: %[[src:.*]] = bitcast <8 x i16> %[[arg1]] to <2 x i64>
! CHECK:  store <2 x i64> %[[src]], ptr %[[addr]], align 1
end subroutine vec_xstd2_test_vi2i4vi2

! CHECK-LABEL: vec_xstd2_test_vi4i4vai4
subroutine vec_xstd2_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %[[arg3:.*]], %[[idx64m1]] : (!fir.ref<!fir.array<20x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[elemptr:.*]] = fir.convert %[[elem]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[bcsrc:.*]] = vector.bitcast %[[vsrc]] : vector<4xi32> to vector<2xi64>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[bcsrc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %[[arg3:.*]][0, %[[idx64m1]]] : (!llvm.ptr<array<20 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[elemptr:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemptr]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[src:.*]] = llvm.bitcast %[[arg1]] : vector<4xi32> to vector<2xi64>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [20 x <4 x i32>], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i32 %[[arg2]]
! CHECK: %[[src:.*]] = bitcast <4 x i32> %[[arg1]] to <2 x i64>
! CHECK: store <2 x i64> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xstd2_test_vi4i4vai4

!----------------------
! vec_xstw4
!----------------------

! CHECK-LABEL: vec_xstw4_test_vr4i2r4
subroutine vec_xstw4_test_vr4i2r4(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  real(4) :: arg3
  call vec_xstw4(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! CHECK-FIR: %[[arg3ptr:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3ptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[vsrc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<i16>
! CHECK-LLVMIR: %[[arg3ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3ptr]][%[[arg2]]] : (!llvm.ptr<i8>, i16) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]
  
! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load i16, ptr %{{.*}}, align 2
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %{{.*}}, i16 %[[arg2]]
! CHECK: store <4 x float> %[[arg1]], ptr %[[addr]], align 1
end subroutine vec_xstw4_test_vr4i2r4

! CHECK-LABEL: vec_xstw4_test_vi4i8ia4
subroutine vec_xstw4_test_vi4i8ia4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(8) :: arg2
  integer(4) :: arg3(10)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %arg2, %[[idx64m1]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK-FIR: %[[elemref:.*]] = fir.convert %[[elem]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemref]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[vsrc]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]]

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i64>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %arg2[0, %[[idx64m1]]] : (!llvm.ptr<array<10 x i32>>, i64) -> !llvm.ptr<i32>
! CHECK-LLVMIR: %[[elemref:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<i32> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemref]][%[[arg2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i64, ptr %1, align 8
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [10 x i32], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i64 %6
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_xstw4_test_vi4i8ia4

! CHECK-LABEL: vec_xstw4_test_vi2i4vi2
subroutine vec_xstw4_test_vi2i4vi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_xstw4(arg1, arg2, arg3)

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[bcsrc:.*]] = vector.bitcast %[[vsrc]] : vector<8xi16> to vector<4xi32>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[bcsrc]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[arg3:.*]] = llvm.bitcast %arg2 : !llvm.ptr<vector<8xi16>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[arg3]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[src:.*]] = llvm.bitcast %[[arg1]] : vector<8xi16> to vector<4xi32>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[src]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! CHECK: %[[src:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32>
! CHECK:  store <4 x i32> %[[src]], ptr %[[addr]], align 1
end subroutine vec_xstw4_test_vi2i4vi2

! CHECK-LABEL: vec_xstw4_test_vi4i4vai4
subroutine vec_xstw4_test_vi4i4vai4(arg1, arg2, arg3, i)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3(20)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! CHECK-FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! CHECK-FIR: %[[idx:.*]] = fir.load %arg3 : !fir.ref<i32>
! CHECK-FIR: %[[idx64:.*]] = fir.convert %[[idx]] : (i32) -> i64
! CHECK-FIR: %[[one:.*]] = arith.constant 1 : i64
! CHECK-FIR: %[[idx64m1:.*]] = arith.subi %[[idx64]], %[[one]] : i64
! CHECK-FIR: %[[elem:.*]] = fir.coordinate_of %[[arg3:.*]], %[[idx64m1]] : (!fir.ref<!fir.array<20x!fir.vector<4:i32>>>, i64) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[elemptr:.*]] = fir.convert %[[elem]] : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! CHECK-FIR: %[[vsrc:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[cnvsrc:.*]] = fir.convert %[[vsrc]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cnvsrc]] to %[[trg]]

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %arg1 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx:.*]] = llvm.load %arg3 : !llvm.ptr<i32>
! CHECK-LLVMIR: %[[idx64:.*]] = llvm.sext %[[idx]] : i32 to i64
! CHECK-LLVMIR: %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
! CHECK-LLVMIR: %[[idx64m1:.*]] = llvm.sub %[[idx64]], %[[one]]  : i64
! CHECK-LLVMIR: %[[elem:.*]] = llvm.getelementptr %[[arg3:.*]][0, %[[idx64m1]]] : (!llvm.ptr<array<20 x vector<4xi32>>>, i64) -> !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[elemptr:.*]] = llvm.bitcast %[[elem]] : !llvm.ptr<vector<4xi32>> to !llvm.ptr<i8>
! CHECK-LLVMIR: %[[addr:.*]] = llvm.getelementptr %[[elemptr]][%[[arg2]]] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
! CHECK-LLVMIR: %[[trg:.*]] = llvm.bitcast %[[addr]] : !llvm.ptr<i8> to !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: llvm.store %[[arg1]], %[[trg]]

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! CHECK: %[[arg2:.*]] = load i32, ptr %1, align 4
! CHECK: %[[idx:.*]] = load i32, ptr %3, align 4
! CHECK: %[[idx64:.*]] = sext i32 %[[idx]] to i64
! CHECK: %[[idx64m1:.*]] = sub i64 %[[idx64]], 1
! CHECK: %[[elem:.*]] = getelementptr [20 x <4 x i32>], ptr %2, i32 0, i64 %[[idx64m1]]
! CHECK: %[[trg:.*]] = getelementptr i8, ptr %[[elem]], i32 %[[arg2]]
! CHECK: store <4 x i32> %[[arg1]], ptr %[[trg]], align 1
end subroutine vec_xstw4_test_vi4i4vai4
