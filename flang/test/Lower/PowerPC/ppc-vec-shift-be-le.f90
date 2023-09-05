! RUN: %flang_fc1 -emit-fir %s -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="CHECK" %s
!
! RUN: %flang_fc1 -emit-fir %s -triple ppc64-unknown-aix -o - | FileCheck --check-prefixes="BE-FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -triple ppc64-unknown-aix -o - | FileCheck --check-prefixes="BE-IR" %s
! REQUIRES: target=powerpc{{.*}}

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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i1

! CHECK-LABEL: vec_sld_test_i2i2
subroutine vec_sld_test_i2i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 8_2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_i2i4

! CHECK-LABEL: vec_sld_test_i2i8
subroutine vec_sld_test_i2i8(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 11_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sld_test_r4i4

! CHECK-LABEL: vec_sld_test_r4i8
subroutine vec_sld_test_r4i8(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_sld(arg1, arg2, 1_8)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[barg2]], %[[barg1]] [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:i16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:i32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:i64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg2]], <16 x i8> %[[arg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! BE-IR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! CHECK: store <8 x i16> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<8xi16> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<8xi16>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! BE-IR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <8 x i16> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <8 x i16> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <8 x i16>
! BE-IR: store <8 x i16> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! CHECK: store <4 x i32> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xi32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xi32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! BE-IR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x i32> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x i32> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x i32>
! BE-IR: store <4 x i32> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! CHECK: store <2 x i64> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xi64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xi64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! BE-IR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x i64> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x i64> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x i64>
! BE-IR: store <2 x i64> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! CHECK: store <4 x float> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<4xf32> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<4xf32>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<4xf32>) -> !fir.vector<4:f32>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! BE-IR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <4 x float> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <4 x float> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <4 x float>
! BE-IR: store <4 x float> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! BE-IR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-IR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! BE-IR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-IR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! BE-IR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-IR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
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

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! CHECK: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[barg2]], <16 x i8> %[[barg1]], <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
! CHECK: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! CHECK: store <2 x double> %[[br]], ptr %{{.*}}, align 16

! BE-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! BE-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! BE-FIR: %[[barg1:.*]] = llvm.bitcast %[[carg1]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[barg2:.*]] = llvm.bitcast %[[carg2]] : vector<2xf64> to vector<16xi8>
! BE-FIR: %[[r:.*]] = vector.shuffle %[[barg1]], %[[barg2]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : vector<16xi8>, vector<16xi8>
! BE-FIR: %[[br:.*]] = llvm.bitcast %[[r]] : vector<16xi8> to vector<2xf64>
! BE-FIR: %[[cr:.*]] = fir.convert %[[br]] : (vector<2xf64>) -> !fir.vector<2:f64>
! BE-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! BE-IR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! BE-IR: %[[barg1:.*]] = bitcast <2 x double> %[[arg1]] to <16 x i8>
! BE-IR: %[[barg2:.*]] = bitcast <2 x double> %[[arg2]] to <16 x i8>
! BE-IR: %[[r:.*]] = shufflevector <16 x i8> %[[barg1]], <16 x i8> %[[barg2]], <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
! BE-IR: %[[br:.*]] = bitcast <16 x i8> %[[r]] to <2 x double>
! BE-IR: store <2 x double> %[[br]], ptr %{{.*}}, align 16
end subroutine vec_sldw_test_r8i8
