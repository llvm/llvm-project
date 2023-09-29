! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -fno-ppc-native-vector-element-order -triple ppc64le-unknown-linux -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: vec_splat_testf32i64
subroutine vec_splat_testf32i64(x)
  vector(real(4)) :: x, y
  y = vec_splat(x, 0_8)

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

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[ele:.*]] = extractelement <16 x i8> %[[x]], i16 15
! LLVMIR: %[[ins:.*]] = insertelement <16 x i8> undef, i8 %[[ele]], i32 0
! LLVMIR: %[[y:.*]] = shufflevector <16 x i8> %[[ins]], <16 x i8> undef, <16 x i32> zeroinitializer
! LLVMIR: store <16 x i8> %[[y]], ptr %{{[0-9]}}, align 16
end subroutine vec_splat_testu8i16
