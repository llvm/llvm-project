! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_st
!----------------------
! CHECK-LABEL: vec_st_test
subroutine vec_st_test(arg1, arg2, arg3)
  vector(integer(2)) :: arg1
  integer(4) :: arg2
  vector(integer(2)) :: arg3
  call vec_st(arg1, arg2, arg3)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! FIR: %[[bc:.*]] = vector.bitcast %[[varg1]] : vector<8xi16> to vector<4xi32>
! FIR: %[[ordr:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shf:.*]] = vector.shuffle %[[bc]], %[[ordr]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: fir.call @llvm.ppc.altivec.stvx(%[[shf]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[bc:.*]] = bitcast <8 x i16> %[[arg1]] to <4 x i32> 
! LLVMIR: %[[shf:.*]] = shufflevector <4 x i32> %[[bc]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  call void @llvm.ppc.altivec.stvx(<4 x i32> %[[shf]], ptr %[[addr]])
end subroutine vec_st_test

!----------------------
! vec_ste
!----------------------
! CHECK-LABEL: vec_ste_test
subroutine vec_ste_test(arg1, arg2, arg3)
  vector(real(4)) :: arg1
  integer(4) :: arg2
  real(4) :: arg3
  call vec_ste(arg1, arg2, arg3)
  
! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[bc:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! FIR: %[[ordr:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shf:.*]] = vector.shuffle %[[bc]], %[[ordr]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: fir.call @llvm.ppc.altivec.stvewx(%[[shf]], %[[addr]]) fastmath<contract> : (vector<4xi32>, !fir.ref<!fir.array<?xi8>>) -> ()

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[bc:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[shf:.*]] = shufflevector <4 x i32> %[[bc]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: call void @llvm.ppc.altivec.stvewx(<4 x i32> %[[shf]], ptr %[[addr]])
end subroutine vec_ste_test

!----------------------
! vec_xst
!----------------------
! CHECK-LABEL: vec_xst_test
subroutine vec_xst_test(arg1, arg2, arg3)
  vector(integer(4)) :: arg1
  integer(4) :: arg2
  vector(integer(4)) :: arg3
  call vec_xst(arg1, arg2, arg3)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[arg3:.*]] = fir.convert %arg2 : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg3]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:i32>>
! FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! FIR: %[[ordr:.*]] = fir.undefined vector<4xi32>
! FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[ordr]] [3, 2, 1, 0] : vector<4xi32>, vector<4xi32>
! FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[trg:.*]] = getelementptr i8, ptr %2, i32 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x i32> %[[arg1]], <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR:  store <4 x i32> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xst_test

!----------------------
! vec_xstd2
!----------------------
! CHECK-LABEL: vec_xstd2_test
subroutine vec_xstd2_test(arg1, arg2, arg3, i)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  vector(real(4)) :: arg3(*)
  integer(4) :: i
  call vec_xstd2(arg1, arg2, arg3(i))

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[arg4:.*]] = fir.load %arg3 : !fir.ref<i32>
! FIR: %[[arg4_64:.*]] = fir.convert %[[arg4]] : (i32) -> i64
! FIR: %[[one:.*]] = arith.constant 1 : i64
! FIR: %[[idx:.*]] = arith.subi %[[arg4_64]], %[[one]] : i64
! FIR: %[[elemaddr:.*]] = fir.coordinate_of %arg2, %[[idx]] : (!fir.ref<!fir.array<?x!fir.vector<4:f32>>>, i64) -> !fir.ref<!fir.vector<4:f32>>
! FIR: %[[elemptr:.*]] = fir.convert %[[elemaddr]] : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[v2elem:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<2xi64>
! FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<2:i64>>
! FIR: %[[undef:.*]] = fir.undefined vector<2xi64>
! FIR: %[[shf:.*]] = vector.shuffle %[[v2elem]], %[[undef]] [1, 0] : vector<2xi64>, vector<2xi64>
! FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[arg4:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[arg4_64:.*]] = sext i32 %[[arg4]] to i64
! LLVMIR: %[[idx:.*]] = sub i64 %[[arg4_64]], 1
! LLVMIR: %[[elemptr:.*]] = getelementptr <4 x float>, ptr %2, i64 %[[idx]]
! LLVMIR: %[[trg:.*]] = getelementptr i8, ptr %[[elemptr]], i16 %[[arg2]]
! LLVMIR: %[[v2elem:.*]] = bitcast <4 x float> %[[arg1]] to <2 x i64>
! LLVMIR: %[[src:.*]] = shufflevector <2 x i64> %[[v2elem]], <2 x i64> undef, <2 x i32> <i32 1, i32 0>
! LLVMIR: store <2 x i64> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xstd2_test

!----------------------
! vec_xstw4
!----------------------
! CHECK-LABEL: vec_xstw4_test
subroutine vec_xstw4_test(arg1, arg2, arg3, i)
  vector(real(4)) :: arg1
  integer(2) :: arg2
  vector(real(4)) :: arg3(*)
  integer(4) :: i
  call vec_xstw4(arg1, arg2, arg3(i))

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[arg2:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[arg4:.*]] = fir.load %arg3 : !fir.ref<i32>
! FIR: %[[arg4_64:.*]] = fir.convert %[[arg4]] : (i32) -> i64
! FIR: %[[one:.*]] = arith.constant 1 : i64
! FIR: %[[idx:.*]] = arith.subi %[[arg4_64]], %[[one]] : i64
! FIR: %[[elemaddr:.*]] = fir.coordinate_of %arg2, %[[idx]] : (!fir.ref<!fir.array<?x!fir.vector<4:f32>>>, i64) -> !fir.ref<!fir.vector<4:f32>>
! FIR: %[[elemptr:.*]] = fir.convert %[[elemaddr]] : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[elemptr]], %[[arg2]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[trg:.*]] = fir.convert %[[addr]] : (!fir.ref<!fir.array<?xi8>>) -> !fir.ref<!fir.vector<4:f32>>
! FIR: %[[undef:.*]] = fir.undefined vector<4xf32>
! FIR: %[[shf:.*]] = vector.shuffle %[[varg1]], %[[undef]] [3, 2, 1, 0] : vector<4xf32>, vector<4xf32>
! FIR: %[[src:.*]] = fir.convert %[[shf]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[src]] to %[[trg]] {alignment = 1 : i64} : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %0, align 16
! LLVMIR: %[[arg2:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[arg4:.*]] = load i32, ptr %3, align 4
! LLVMIR: %[[arg4_64:.*]] = sext i32 %[[arg4]] to i64
! LLVMIR: %[[idx:.*]] = sub i64 %[[arg4_64]], 1
! LLVMIR: %[[elemptr:.*]] = getelementptr <4 x float>, ptr %2, i64 %[[idx]]
! LLVMIR: %[[trg:.*]] = getelementptr i8, ptr %[[elemptr]], i16 %[[arg2]]
! LLVMIR: %[[src:.*]] = shufflevector <4 x float> %[[arg1]], <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
! LLVMIR: store <4 x float> %[[src]], ptr %[[trg]], align 1
end subroutine vec_xstw4_test
