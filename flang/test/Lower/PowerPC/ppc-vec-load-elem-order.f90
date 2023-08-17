! RUN: bbc -emit-fir %s -fno-ppc-native-vector-element-order -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!-------------------
! vec_ld
!-------------------

! CHECK-LABEL: @vec_ld_testi8
subroutine vec_ld_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<16:i8>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<16xi8>
! FIR: %[[undefv:.*]] = fir.undefined vector<16xi8>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xi8>, vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[bc]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi8

! CHECK-LABEL: @vec_ld_testi16
subroutine vec_ld_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<8xi16>
! FIR: %[[undefv:.*]] = fir.undefined vector<8xi16>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[bc]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi16

! CHECK-LABEL: @vec_ld_testi32
subroutine vec_ld_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32

! CHECK-LABEL: @vec_ld_testf32
subroutine vec_ld_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[i4:.*]] = fir.convert %[[arg1]] : (i64) -> i32
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[i4]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[i4:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[i4]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testf32

! CHECK-LABEL: @vec_ld_testu32
subroutine vec_ld_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<4:ui32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:ui32>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testu32

! CHECK-LABEL: @vec_ld_testi32a
subroutine vec_ld_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32a

! CHECK-LABEL: @vec_ld_testf32av
subroutine vec_ld_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[i4:.*]] = fir.convert %[[arg1]] : (i64) -> i32
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[i4]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[i4:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[i4]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testf32av

! CHECK-LABEL: @vec_ld_testi32s
subroutine vec_ld_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_ld_testi32s

!-------------------
! vec_lde
!-------------------

! CHECK-LABEL: @vec_lde_testi8s
subroutine vec_lde_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvebx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[undefv:.*]] = fir.undefined vector<16xi8>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xi8>, vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvebx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi8s

! CHECK-LABEL: @vec_lde_testi16a
subroutine vec_lde_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 11, 7)
  vector(integer(2)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x11x7xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvehx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<8xi16>
! FIR: %[[undefv:.*]] = fir.undefined vector<8xi16>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <8 x i16> @llvm.ppc.altivec.lvehx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[ld]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi16a

! CHECK-LABEL: @vec_lde_testi32a
subroutine vec_lde_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(5)
  vector(integer(4)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<5xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvewx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ld]], %[[undefv]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x i32> %[[ld]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testi32a

! CHECK-LABEL: @vec_lde_testf32a
subroutine vec_lde_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(11)
  vector(real(4)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<11xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvewx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[bc]], %[[undefv]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: %[[shflv:.*]] = shufflevector <4 x float> %[[bc]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[shflv]], ptr %2, align 16
end subroutine vec_lde_testf32a

!-------------------
! vec_xld2
!-------------------

! CHECK-LABEL: @vec_xld2_testi8a
subroutine vec_xld2_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<16:i8>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi8a

! CHECK-LABEL: @vec_xld2_testi16a
subroutine vec_xld2_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(4)
  vector(integer(2)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<8:i16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <8 x i16>
! LLVMIR:  store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi16a

! CHECK-LABEL: @vec_xld2_testi32a
subroutine vec_xld2_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2(11)
  vector(integer(4)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<11x!fir.vector<4:i32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi32a

! CHECK-LABEL: @vec_xld2_testi64a
subroutine vec_xld2_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(31,7)
  vector(integer(8)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<31x7x!fir.vector<2:i64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<2xi64>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi64a

! CHECK-LABEL: @vec_xld2_testf32a
subroutine vec_xld2_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(5)
  vector(real(4)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<5x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testf32a

! CHECK-LABEL: @vec_xld2_testf64a
subroutine vec_xld2_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(8)) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<2:f64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xld2_testf64a

!-------------------
! vec_xlw4
!-------------------

! CHECK-LABEL: @vec_xlw4_testi8a
subroutine vec_xlw4_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(2, 11, 37)
  vector(integer(1)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x11x37x!fir.vector<16:i8>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testi8a

! CHECK-LABEL: @vec_xlw4_testi16a
subroutine vec_xlw4_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(2, 8)
  vector(integer(2)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x8x!fir.vector<8:i16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testi16a

! CHECK-LABEL: @vec_xlw4_testu32a
subroutine vec_xlw4_testu32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(unsigned(4)) :: arg2(8, 4)
  vector(unsigned(4)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<8x4x!fir.vector<4:ui32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:ui32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xlw4_testu32a

! CHECK-LABEL: @vec_xlw4_testf32a
subroutine vec_xlw4_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2
  vector(real(4)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xlw4_testf32a
