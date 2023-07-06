! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_any_ge
!----------------------

! CHECK-LABEL: vec_any_ge_test_i1
subroutine vec_any_ge_test_i1(arg1, arg2)
  vector(integer(1)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsb.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<16:i8>, !fir.vector<16:i8>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtsb.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<16xi8>, vector<16xi8>) -> i32

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsb.p(i32 3, <16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_any_ge_test_i1

! CHECK-LABEL: vec_any_ge_test_i2
subroutine vec_any_ge_test_i2(arg1, arg2)
  vector(integer(2)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsh.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<8:i16>, !fir.vector<8:i16>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtsh.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<8xi16>, vector<8xi16>) -> i32

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsh.p(i32 3, <8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_any_ge_test_i2

! CHECK-LABEL: vec_any_ge_test_i4
subroutine vec_any_ge_test_i4(arg1, arg2)
  vector(integer(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsw.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<4:i32>, !fir.vector<4:i32>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtsw.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<4xi32>, vector<4xi32>) -> i32

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsw.p(i32 3, <4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_any_ge_test_i4

! CHECK-LABEL: vec_any_ge_test_i8
subroutine vec_any_ge_test_i8(arg1, arg2)
  vector(integer(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtsd.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<2:i64>, !fir.vector<2:i64>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtsd.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<2xi64>, vector<2xi64>) -> i32

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtsd.p(i32 3, <2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_any_ge_test_i8

! CHECK-LABEL: vec_any_ge_test_u1
subroutine vec_any_ge_test_u1(arg1, arg2)
  vector(unsigned(1)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtub.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<16:ui8>, !fir.vector<16:ui8>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtub.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<16xi8>, vector<16xi8>) -> i32

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtub.p(i32 3, <16 x i8> %[[arg2]], <16 x i8> %[[arg1]])
end subroutine vec_any_ge_test_u1

! CHECK-LABEL: vec_any_ge_test_u2
subroutine vec_any_ge_test_u2(arg1, arg2)
  vector(unsigned(2)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuh.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<8:ui16>, !fir.vector<8:ui16>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtuh.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<8xi16>, vector<8xi16>) -> i32

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtuh.p(i32 3, <8 x i16> %[[arg2]], <8 x i16> %[[arg1]])
end subroutine vec_any_ge_test_u2

! CHECK-LABEL: vec_any_ge_test_u4
subroutine vec_any_ge_test_u4(arg1, arg2)
  vector(unsigned(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtuw.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<4:ui32>, !fir.vector<4:ui32>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtuw.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<4xi32>, vector<4xi32>) -> i32

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtuw.p(i32 3, <4 x i32> %[[arg2]], <4 x i32> %[[arg1]])
end subroutine vec_any_ge_test_u4

! CHECK-LABEL: vec_any_ge_test_u8
subroutine vec_any_ge_test_u8(arg1, arg2)
  vector(unsigned(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[op:.*]] = arith.constant 3 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.altivec.vcmpgtud.p(%[[op]], %[[arg2]], %[[arg1]]) fastmath<contract> : (i32, !fir.vector<2:ui64>, !fir.vector<2:ui64>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(3 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.altivec.vcmpgtud.p(%[[op]], %[[arg2]], %[[arg1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<2xi64>, vector<2xi64>) -> i32

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.altivec.vcmpgtud.p(i32 3, <2 x i64> %[[arg2]], <2 x i64> %[[arg1]])
end subroutine vec_any_ge_test_u8

! CHECK-LABEL: vec_any_ge_test_r4
subroutine vec_any_ge_test_r4(arg1, arg2)
  vector(real(4)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[op:.*]] = arith.constant 1 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgesp.p(%[[op]], %[[arg1]], %[[arg2]]) fastmath<contract> : (i32, !fir.vector<4:f32>, !fir.vector<4:f32>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(1 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.vsx.xvcmpgesp.p(%[[op]], %[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<4xf32>, vector<4xf32>) -> i32

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.vsx.xvcmpgesp.p(i32 1, <4 x float> %[[arg1]], <4 x float> %[[arg2]])
end subroutine vec_any_ge_test_r4

! CHECK-LABEL: vec_any_ge_test_r8
subroutine vec_any_ge_test_r8(arg1, arg2)
  vector(real(8)), intent(in) :: arg1, arg2
  integer(4) :: r
  r = vec_any_ge(arg1, arg2)

! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[op:.*]] = arith.constant 1 : i32
! CHECK-FIR: %{{[0-9]+}} = fir.call @llvm.ppc.vsx.xvcmpgedp.p(%[[op]], %[[arg1]], %[[arg2]]) fastmath<contract> : (i32, !fir.vector<2:f64>, !fir.vector<2:f64>) -> i32

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[op:.*]] = llvm.mlir.constant(1 : i32) : i32
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.call @llvm.ppc.vsx.xvcmpgedp.p(%[[op]], %[[arg1]], %[[arg2]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, vector<2xf64>, vector<2xf64>) -> i32

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = call i32 @llvm.ppc.vsx.xvcmpgedp.p(i32 1, <2 x double> %[[arg1]], <2 x double> %[[arg2]])
end subroutine vec_any_ge_test_r8

