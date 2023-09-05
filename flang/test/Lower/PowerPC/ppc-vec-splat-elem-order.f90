! RUN: %flang_fc1 -emit-fir %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_splat_testf32i64
subroutine vec_splat_testf32i64(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_8)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! FIR: %[[idx:.*]] = arith.constant 0 : i64
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! FIR: %[[c:.*]] = arith.constant 4 : i64
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i64
! FIR: %[[c2:.*]] = arith.constant 3 : i64
! FIR: %[[sub:.*]] = llvm.sub %[[c2]], %[[u]]  : i64
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[sub]] : i64] : vector<4xf32>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<4xf32>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <4 x float> %[[x]], i64 3
! LLVMIR: %[[ins:.*]] = insertelement <4 x float> undef, float %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <4 x float> %[[ins]], <4 x float> undef, <4 x i32> zeroinitializer
! LLVMIR: store <4 x float> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testf32i64

! CHECK-LABEL: vec_splat_testu8i16
subroutine vec_splat_testu8i16(x)
  vector(unsigned(1)) :: x, y
  y = vec_splat(x, 0_2)
! FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! FIR: %[[idx:.*]] = arith.constant 0 : i16
! FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! FIR: %[[c:.*]] = arith.constant 16 : i16
! FIR: %[[u:.*]] = llvm.urem %[[idx]], %[[c]]  : i16
! FIR: %[[c2:.*]] = arith.constant 15 : i16
! FIR: %[[sub:.*]] = llvm.sub %[[c2]], %[[u]]  : i16
! FIR: %[[ele:.*]] = vector.extractelement %[[vx]][%[[sub]] : i16] : vector<16xi8>
! FIR: %[[vy:.*]] = vector.splat %[[ele]] : vector<16xi8>
! FIR: %[[y:.*]] = fir.convert %[[vy]] : (vector<16xi8>) -> !fir.vector<16:ui8>
! FIR: fir.store %[[y]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i16 15
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i16
