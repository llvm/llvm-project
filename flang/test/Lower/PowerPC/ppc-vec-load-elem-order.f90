! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
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
! vec_lvsl
!-------------------

! CHECK-LABEL: @vec_lvsl_testi8s
subroutine vec_lvsl_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i8) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[iext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi8s

! CHECK-LABEL: @vec_lvsl_testi16a
subroutine vec_lvsl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i16) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[iext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi16a

! CHECK-LABEL: @vec_lvsl_testi32a
subroutine vec_lvsl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(11, 3, 4)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i32) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<11x3x4xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testi32a

! CHECK-LABEL: @vec_lvsl_testf32a
subroutine vec_lvsl_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(51)
  vector(unsigned(1)) :: res
  res = vec_lvsl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<51xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsl(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsl_testf32a

!-------------------
! vec_lvsr
!-------------------

! CHECK-LABEL: @vec_lvsr_testi8s
subroutine vec_lvsr_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i8) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsr(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[iext:.*]] = sext i8 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi8s

! CHECK-LABEL: @vec_lvsr_testi16a
subroutine vec_lvsr_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(41)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i16) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<41xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsr(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[iext:.*]] = sext i16 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi16a

! CHECK-LABEL: @vec_lvsr_testi32a
subroutine vec_lvsr_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(23, 31, 47)
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg1i64:.*]] = fir.convert %[[arg1]] : (i32) -> i64
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1i64]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<23x31x47xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsr(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[iext:.*]] = sext i32 %[[arg1]] to i64
! LLVMIR: %[[lshft:.*]] = shl i64 %[[iext]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testi32a

! CHECK-LABEL: @vec_lvsr_testf32a
subroutine vec_lvsr_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2
  vector(unsigned(1)) :: res
  res = vec_lvsr(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[fiveSix:.*]] = arith.constant 56 : i64
! FIR: %[[lshft:.*]] = arith.shli %[[arg1]], %[[fiveSix]] : i64
! FIR: %[[rshft:.*]] = arith.shrsi %[[lshft]], %[[fiveSix]] : i64
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[rshft]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.altivec.lvsr(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[lshft:.*]] = shl i64 %[[arg1]], 56
! LLVMIR: %[[rshft:.*]] = ashr i64 %[[lshft]], 56
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[rshft]]
! LLVMIR: %[[ld:.*]] = call <16 x i8> @llvm.ppc.altivec.lvsr(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[ld]], ptr %2, align 16
end subroutine vec_lvsr_testf32a

!-------------------
! vec_lxv
!-------------------

! CHECK-LABEL: @vec_lxv_testi8a
subroutine vec_lxv_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_lxv(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[offset:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[offset]]
! LLVMIR: %[[res:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: store <16 x i8> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi8a

! CHECK-LABEL: @vec_lxv_testi16a
subroutine vec_lxv_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_lxv(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[offset:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[offset]]
! LLVMIR: %[[res:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: store <8 x i16> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi16a

! CHECK-LABEL: @vec_lxv_testi32a
subroutine vec_lxv_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_lxv(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[offset:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[offset]]
! LLVMIR: %[[res:.*]] = load <4 x i32>, ptr %[[addr]], align 1
! LLVMIR: store <4 x i32> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testi32a

! CHECK-LABEL: @vec_lxv_testf32a
subroutine vec_lxv_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_lxv(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[offset:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[offset]]
! LLVMIR: %[[res:.*]] = load <4 x float>, ptr %[[addr]], align 1
! LLVMIR: store <4 x float> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testf32a

! CHECK-LABEL: @vec_lxv_testf64a
subroutine vec_lxv_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(4)
  vector(real(8)) :: res
  res = vec_lxv(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[offset:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[offset]]
! LLVMIR: %[[res:.*]] = load <2 x double>, ptr %[[addr]], align 1
! LLVMIR: store <2 x double> %[[res]], ptr %2, align 16
end subroutine vec_lxv_testf64a

!-------------------
! vec_xl
!-------------------

! CHECK-LABEL: @vec_xl_testi8a
subroutine vec_xl_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<16xi8>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xi8>, vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>
  
! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <16 x i8>, ptr %[[addr]], align 1
! LLVMIR: %[[shflv:.*]] = shufflevector <16 x i8> %[[ld]], <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %[[shflv]], ptr %2, align 16
end subroutine vec_xl_testi8a

! CHECK-LABEL: @vec_xl_testi16a
subroutine vec_xl_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 8)
  vector(integer(2)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x8xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<8xi16>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load <8 x i16>, ptr %[[addr]], align 1
! LLVMIR: %[[shflv:.*]] = shufflevector <8 x i16> %[[ld]], <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %[[shflv]], ptr %2, align 16
end subroutine vec_xl_testi16a

! CHECK-LABEL: @vec_xl_testi32a
subroutine vec_xl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4, 8)
  vector(integer(4)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testi32a

! CHECK-LABEL: @vec_xl_testi64a
subroutine vec_xl_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 1)
  vector(integer(8)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x1xi64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<2xi64>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xl_testi64a

! CHECK-LABEL: @vec_xl_testf32a
subroutine vec_xl_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x.be(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xl_testf32a

! CHECK-LABEL: @vec_xl_testf64a
subroutine vec_xl_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(2)
  vector(real(8)) :: res
  res = vec_xl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2xf64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x.be(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x.be(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xl_testf64a

!-------------------
! vec_xl_be
!-------------------

! CHECK-LABEL: @vec_xl_be_testi8a
subroutine vec_xl_be_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2(2, 4, 8)
  vector(integer(1)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi8>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<16xi8>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<16xi8>, vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>
  
! LLVMIR: %4 = load i8, ptr %0, align 1
! LLVMIR: %5 = getelementptr i8, ptr %1, i8 %4
! LLVMIR: %6 = load <16 x i8>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <16 x i8> %7, ptr %2, align 16
end subroutine vec_xl_be_testi8a

! CHECK-LABEL: @vec_xl_be_testi16a
subroutine vec_xl_be_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(8,2)
  vector(integer(2)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<8x2xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<8xi16>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %4 = load i16, ptr %0, align 2
! LLVMIR: %5 = getelementptr i8, ptr %1, i16 %4
! LLVMIR: %6 = load <8 x i16>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <8 x i16> %6, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <8 x i16> %7, ptr %2, align 16
end subroutine vec_xl_be_testi16a

! CHECK-LABEL: @vec_xl_be_testi32a
subroutine vec_xl_be_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(2, 4)
  vector(integer(4)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %4 = load i32, ptr %0, align 4
! LLVMIR: %5 = getelementptr i8, ptr %1, i32 %4
! LLVMIR: %6 = load <4 x i32>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <4 x i32> %6, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x i32> %7, ptr %2, align 16
end subroutine vec_xl_be_testi32a

! CHECK-LABEL: @vec_xl_be_testi64a
subroutine vec_xl_be_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  integer(8) :: arg2(2, 4, 8)
  vector(integer(8)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<2xi64>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [1, 0] : vector<2xi64>, vector<2xi64>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %4 = load i64, ptr %0, align 8
! LLVMIR: %5 = getelementptr i8, ptr %1, i64 %4
! LLVMIR: %6 = load <2 x i64>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <2 x i64> %6, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x i64> %7, ptr %2, align 16
end subroutine vec_xl_be_testi64a

! CHECK-LABEL: @vec_xl_be_testf32a
subroutine vec_xl_be_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %4 = load i16, ptr %0, align 2
! LLVMIR: %5 = getelementptr i8, ptr %1, i16 %4
! LLVMIR: %6 = load <4 x float>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <4 x float> %6, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %7, ptr %2, align 16
end subroutine vec_xl_be_testf32a

! CHECK-LABEL: @vec_xl_be_testf64a
subroutine vec_xl_be_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  real(8) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xl_be(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[ref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[ref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref2:.*]] = fir.load %[[addr]] {alignment = 1 : i64} : !fir.ref<!fir.array<?xi8>>
! FIR: %[[undefv:.*]] = fir.undefined vector<2xf64>
! FIR: %[[shflv:.*]] = vector.shuffle %[[ref2]], %[[undefv]] [1, 0] : vector<2xf64>, vector<2xf64>
! FIR: %[[res:.*]] = fir.convert %[[shflv]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %4 = load i64, ptr %0, align 8
! LLVMIR: %5 = getelementptr i8, ptr %1, i64 %4
! LLVMIR: %6 = load <2 x double>, ptr %5, align 1
! LLVMIR: %7 = shufflevector <2 x double> %6, <2 x double> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x double> %7, ptr %2, align 16
end subroutine vec_xl_be_testf64a

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

!-------------------
! vec_xlds
!-------------------

! CHECK-LABEL: @vec_xlds_testi64a
subroutine vec_xlds_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(4)
  vector(integer(8)) :: res
  res = vec_xlds(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[aryref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<2:i64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[aryref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<i64>
! FIR: %[[val:.*]] = fir.load %[[ref]] : !fir.ref<i64>
! FIR: %[[vsplt:.*]] = vector.splat %[[val]] : vector<2xi64>
! FIR: %[[res:.*]] = fir.convert %[[vsplt]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load i64, ptr %[[addr]], align 8
! LLVMIR: %[[insrt:.*]] = insertelement <2 x i64> undef, i64 %[[ld]], i32 0
! LLVMIR: %[[shflv:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: store <2 x i64> %[[shflv]], ptr %2, align 16
end subroutine vec_xlds_testi64a

! CHECK-LABEL: @vec_xlds_testf64a
subroutine vec_xlds_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(8)) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xlds(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[aryref:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<2:f64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[aryref]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ref:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<i64>
! FIR: %[[val:.*]] = fir.load %[[ref]] : !fir.ref<i64>
! FIR: %[[vsplt:.*]] = vector.splat %[[val]] : vector<2xi64>
! FIR: %[[bc:.*]] = vector.bitcast %[[vsplt]] : vector<2xi64> to vector<2xf64>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = load i64, ptr %[[addr]], align 8
! LLVMIR: %[[insrt:.*]] = insertelement <2 x i64> undef, i64 %[[ld]], i32 0
! LLVMIR: %[[shflv:.*]] = shufflevector <2 x i64> %[[insrt]], <2 x i64> undef, <2 x i32> zeroinitializer
! LLVMIR: %[[bc:.*]] = bitcast <2 x i64> %[[shflv]] to <2 x double>
! LLVMIR: store <2 x double> %[[bc]], ptr %2, align 16
end subroutine vec_xlds_testf64a
