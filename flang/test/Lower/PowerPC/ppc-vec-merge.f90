! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!------------
! vec_mergeh
!------------

  ! CHECK-LABEL: vec_mergeh_test_i1
subroutine vec_mergeh_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i1

! CHECK-LABEL: vec_mergeh_test_i2
subroutine vec_mergeh_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 8, 1, 9, 2, 10, 3, 11] : vector<8xi16>, vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 8, 1, 9, 2, 10, 3, 11] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i2

! CHECK-LABEL: vec_mergeh_test_i4
subroutine vec_mergeh_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 4, 1, 5] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 4, 1, 5] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i4

! CHECK-LABEL: vec_mergeh_test_i8
subroutine vec_mergeh_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_i8

! CHECK-LABEL: vec_mergeh_test_u1
subroutine vec_mergeh_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u1

! CHECK-LABEL: vec_mergeh_test_u2
subroutine vec_mergeh_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 8, 1, 9, 2, 10, 3, 11] : vector<8xi16>, vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 8, 1, 9, 2, 10, 3, 11] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u2

! CHECK-LABEL: vec_mergeh_test_u4
subroutine vec_mergeh_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 4, 1, 5] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 4, 1, 5] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u4

! CHECK-LABEL: vec_mergeh_test_u8
subroutine vec_mergeh_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 2] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_u8

! CHECK-LABEL: vec_mergeh_test_r4
subroutine vec_mergeh_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 4, 1, 5] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> %[[arg2]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_r4

! CHECK-LABEL: vec_mergeh_test_r8
subroutine vec_mergeh_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_mergeh(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [0, 2] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [0, 2] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 0, i32 2>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergeh_test_r8

!------------
! vec_mergel
!------------

! CHECK-LABEL: vec_mergel_test_i1
subroutine vec_mergel_test_i1(arg1, arg2)
  vector(integer(1)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i1

! CHECK-LABEL: vec_mergel_test_i2
subroutine vec_mergel_test_i2(arg1, arg2)
  vector(integer(2)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [4, 12, 5, 13, 6, 14, 7, 15] : vector<8xi16>, vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [4, 12, 5, 13, 6, 14, 7, 15] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i2

! CHECK-LABEL: vec_mergel_test_i4
subroutine vec_mergel_test_i4(arg1, arg2)
  vector(integer(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [2, 6, 3, 7] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [2, 6, 3, 7] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i4

! CHECK-LABEL: vec_mergel_test_i8
subroutine vec_mergel_test_i8(arg1, arg2)
  vector(integer(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_i8

! CHECK-LABEL: vec_mergel_test_u1
subroutine vec_mergel_test_u1(arg1, arg2)
  vector(unsigned(1)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xi8>, vector<16xi8>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <16 x i8> %[[arg1]], <16 x i8> %[[arg2]], <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
! CHECK: store <16 x i8> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u1

! CHECK-LABEL: vec_mergel_test_u2
subroutine vec_mergel_test_u2(arg1, arg2)
  vector(unsigned(2)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [4, 12, 5, 13, 6, 14, 7, 15] : vector<8xi16>, vector<8xi16>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [4, 12, 5, 13, 6, 14, 7, 15] : vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <8 x i16> %[[arg1]], <8 x i16> %[[arg2]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
! CHECK: store <8 x i16> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u2

! CHECK-LABEL: vec_mergel_test_u4
subroutine vec_mergel_test_u4(arg1, arg2)
  vector(unsigned(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [2, 6, 3, 7] : vector<4xi32>, vector<4xi32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [2, 6, 3, 7] : vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! CHECK: store <4 x i32> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u4

! CHECK-LABEL: vec_mergel_test_u8
subroutine vec_mergel_test_u8(arg1, arg2)
  vector(unsigned(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 3] : vector<2xi64>, vector<2xi64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x i64> %[[arg1]], <2 x i64> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x i64> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_u8

! CHECK-LABEL: vec_mergel_test_r4
subroutine vec_mergel_test_r4(arg1, arg2)
  vector(real(4)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [2, 6, 3, 7] : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> %[[arg2]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
! CHECK: store <4 x float> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_r4

! CHECK-LABEL: vec_mergel_test_r8
subroutine vec_mergel_test_r8(arg1, arg2)
  vector(real(8)) :: arg1, arg2, r
  r = vec_mergel(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[carg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[carg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[r:.*]] = vector.shuffle %[[carg1]], %[[carg2]] [1, 3] : vector<2xf64>, vector<2xf64>
! CHECK-FIR: %[[cr:.*]] = fir.convert %[[r]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[cr]] to %{{.*}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[r:.*]] = llvm.shufflevector %[[arg1]], %[[arg2]] [1, 3] : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[r]], %{{.*}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[r:.*]] = shufflevector <2 x double> %[[arg1]], <2 x double> %[[arg2]], <2 x i32> <i32 1, i32 3>
! CHECK: store <2 x double> %[[r]], ptr %{{.*}}, align 16
end subroutine vec_mergel_test_r8
