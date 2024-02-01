! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! vec_add

! CHECK-LABEL: vec_add_testf32
subroutine vec_add_testf32(x, y)
  vector(real(4)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fadd contract <4 x float> %[[x]], %[[y]]
end subroutine vec_add_testf32

! CHECK-LABEL: vec_add_testf64
subroutine vec_add_testf64(x, y)
  vector(real(8)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fadd contract <2 x double> %[[x]], %[[y]]
end subroutine vec_add_testf64

! CHECK-LABEL: vec_add_testi8
subroutine vec_add_testi8(x, y)
  vector(integer(1)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <16 x i8> %[[x]], %[[y]]
end subroutine vec_add_testi8

! CHECK-LABEL: vec_add_testi16
subroutine vec_add_testi16(x, y)
  vector(integer(2)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <8 x i16> %[[x]], %[[y]]
end subroutine vec_add_testi16

! CHECK-LABEL: vec_add_testi32
subroutine vec_add_testi32(x, y)
  vector(integer(4)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <4 x i32> %[[x]], %[[y]]
end subroutine vec_add_testi32

! CHECK-LABEL: vec_add_testi64
subroutine vec_add_testi64(x, y)
  vector(integer(8)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <2 x i64> %[[x]], %[[y]]
end subroutine vec_add_testi64

! CHECK-LABEL: vec_add_testui8
subroutine vec_add_testui8(x, y)
  vector(unsigned(1)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <16 x i8> %[[x]], %[[y]]
end subroutine vec_add_testui8

! CHECK-LABEL: vec_add_testui16
subroutine vec_add_testui16(x, y)
  vector(unsigned(2)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <8 x i16> %[[x]], %[[y]]
end subroutine vec_add_testui16

! CHECK-LABEL: vec_add_testui32
subroutine vec_add_testui32(x, y)
  vector(unsigned(4)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <4 x i32> %[[x]], %[[y]]
end subroutine vec_add_testui32

! CHECK-LABEL: vec_add_testui64
subroutine vec_add_testui64(x, y)
  vector(unsigned(8)) :: vsum, x, y
  vsum = vec_add(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = add <2 x i64> %[[x]], %[[y]]
end subroutine vec_add_testui64

! vec_mul

! CHECK-LABEL: vec_mul_testf32
subroutine vec_mul_testf32(x, y)
  vector(real(4)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fmul contract <4 x float> %[[x]], %[[y]]
end subroutine vec_mul_testf32

! CHECK-LABEL: vec_mul_testf64
subroutine vec_mul_testf64(x, y)
  vector(real(8)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fmul contract <2 x double> %[[x]], %[[y]]
end subroutine vec_mul_testf64

! CHECK-LABEL: vec_mul_testi8
subroutine vec_mul_testi8(x, y)
  vector(integer(1)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <16 x i8> %[[x]], %[[y]]
end subroutine vec_mul_testi8

! CHECK-LABEL: vec_mul_testi16
subroutine vec_mul_testi16(x, y)
  vector(integer(2)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <8 x i16> %[[x]], %[[y]]
end subroutine vec_mul_testi16

! CHECK-LABEL: vec_mul_testi32
subroutine vec_mul_testi32(x, y)
  vector(integer(4)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <4 x i32> %[[x]], %[[y]]
end subroutine vec_mul_testi32

! CHECK-LABEL: vec_mul_testi64
subroutine vec_mul_testi64(x, y)
  vector(integer(8)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <2 x i64> %[[x]], %[[y]]
end subroutine vec_mul_testi64

! CHECK-LABEL: vec_mul_testui8
subroutine vec_mul_testui8(x, y)
  vector(unsigned(1)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <16 x i8> %[[x]], %[[y]]
end subroutine vec_mul_testui8

! CHECK-LABEL: vec_mul_testui16
subroutine vec_mul_testui16(x, y)
  vector(unsigned(2)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <8 x i16> %[[x]], %[[y]]
end subroutine vec_mul_testui16

! CHECK-LABEL: vec_mul_testui32
subroutine vec_mul_testui32(x, y)
  vector(unsigned(4)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <4 x i32> %[[x]], %[[y]]
end subroutine vec_mul_testui32

! CHECK-LABEL: vec_mul_testui64
subroutine vec_mul_testui64(x, y)
  vector(unsigned(8)) :: vmul, x, y
  vmul = vec_mul(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = mul <2 x i64> %[[x]], %[[y]]
end subroutine vec_mul_testui64

! vec_sub

! CHECK-LABEL: vec_sub_testf32
subroutine vec_sub_testf32(x, y)
  vector(real(4)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fsub contract <4 x float> %[[x]], %[[y]]
end subroutine vec_sub_testf32

! CHECK-LABEL: vec_sub_testf64
subroutine vec_sub_testf64(x, y)
  vector(real(8)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = fsub contract <2 x double> %[[x]], %[[y]]
end subroutine vec_sub_testf64

! CHECK-LABEL: vec_sub_testi8
subroutine vec_sub_testi8(x, y)
  vector(integer(1)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <16 x i8> %[[x]], %[[y]]
end subroutine vec_sub_testi8

! CHECK-LABEL: vec_sub_testi16
subroutine vec_sub_testi16(x, y)
  vector(integer(2)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <8 x i16> %[[x]], %[[y]]
end subroutine vec_sub_testi16

! CHECK-LABEL: vec_sub_testi32
subroutine vec_sub_testi32(x, y)
  vector(integer(4)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <4 x i32> %[[x]], %[[y]]
end subroutine vec_sub_testi32

! CHECK-LABEL: vec_sub_testi64
subroutine vec_sub_testi64(x, y)
  vector(integer(8)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <2 x i64> %[[x]], %[[y]]
end subroutine vec_sub_testi64

! CHECK-LABEL: vec_sub_testui8
subroutine vec_sub_testui8(x, y)
  vector(unsigned(1)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <16 x i8> %[[x]], %[[y]]
end subroutine vec_sub_testui8

! CHECK-LABEL: vec_sub_testui16
subroutine vec_sub_testui16(x, y)
  vector(unsigned(2)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <8 x i16> %[[x]], %[[y]]
end subroutine vec_sub_testui16

! CHECK-LABEL: vec_sub_testui32
subroutine vec_sub_testui32(x, y)
  vector(unsigned(4)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <4 x i32> %[[x]], %[[y]]
end subroutine vec_sub_testui32

! CHECK-LABEL: vec_sub_testui64
subroutine vec_sub_testui64(x, y)
  vector(unsigned(8)) :: vsub, x, y
  vsub = vec_sub(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %{{[0-9]}} = sub <2 x i64> %[[x]], %[[y]]
end subroutine vec_sub_testui64

!----------------------
! vec_and
!----------------------

! CHECK-LABEL: vec_and_test_i8
subroutine vec_and_test_i8(arg1, arg2)
  vector(integer(1)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i8

! CHECK-LABEL: vec_and_test_i16
subroutine vec_and_test_i16(arg1, arg2)
  vector(integer(2)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i16

! CHECK-LABEL: vec_and_test_i32
subroutine vec_and_test_i32(arg1, arg2)
  vector(integer(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i32

! CHECK-LABEL: vec_and_test_i64
subroutine vec_and_test_i64(arg1, arg2)
  vector(integer(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i64

! CHECK-LABEL: vec_and_test_u8
subroutine vec_and_test_u8(arg1, arg2)
  vector(unsigned(1)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u8

! CHECK-LABEL: vec_and_test_u16
subroutine vec_and_test_u16(arg1, arg2)
  vector(unsigned(2)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u16

! CHECK-LABEL: vec_and_test_u32
subroutine vec_and_test_u32(arg1, arg2)
  vector(unsigned(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u32

! CHECK-LABEL: vec_and_test_u64
subroutine vec_and_test_u64(arg1, arg2)
  vector(unsigned(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = and <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u64

! CHECK-LABEL: vec_and_testf32
subroutine vec_and_testf32(arg1, arg2)
  vector(real(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[bc2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! LLVMIR: %[[r:.*]] = and <4 x i32> %[[bc1]], %[[bc2]]
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[r]] to <4 x float>
end subroutine vec_and_testf32

! CHECK-LABEL: vec_and_testf64
subroutine vec_and_testf64(arg1, arg2)
  vector(real(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <2 x double> %[[arg1]] to <2 x i64>
! LLVMIR: %[[bc2:.*]] = bitcast <2 x double> %[[arg2]] to <2 x i64>
! LLVMIR: %[[r:.*]] = and <2 x i64> %[[bc1]], %[[bc2]]
! LLVMIR: %{{[0-9]+}} = bitcast <2 x i64> %[[r]] to <2 x double>
end subroutine vec_and_testf64

!----------------------
! vec_xor
!----------------------

! CHECK-LABEL: vec_xor_test_i8
subroutine vec_xor_test_i8(arg1, arg2)
  vector(integer(1)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i8

! CHECK-LABEL: vec_xor_test_i16
subroutine vec_xor_test_i16(arg1, arg2)
  vector(integer(2)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i16

! CHECK-LABEL: vec_xor_test_i32
subroutine vec_xor_test_i32(arg1, arg2)
  vector(integer(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i32

! CHECK-LABEL: vec_xor_test_i64
subroutine vec_xor_test_i64(arg1, arg2)
  vector(integer(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i64

! CHECK-LABEL: vec_xor_test_u8
subroutine vec_xor_test_u8(arg1, arg2)
  vector(unsigned(1)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u8

! CHECK-LABEL: vec_xor_test_u16
subroutine vec_xor_test_u16(arg1, arg2)
  vector(unsigned(2)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u16

! CHECK-LABEL: vec_xor_test_u32
subroutine vec_xor_test_u32(arg1, arg2)
  vector(unsigned(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u32

! CHECK-LABEL: vec_xor_test_u64
subroutine vec_xor_test_u64(arg1, arg2)
  vector(unsigned(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %{{[0-9]+}} = xor <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u64

! CHECK-LABEL: vec_xor_testf32
subroutine vec_xor_testf32(arg1, arg2)
  vector(real(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! LLVMIR: %[[bc2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! LLVMIR: %[[r:.*]] = xor <4 x i32> %[[bc1]], %[[bc2]]
! LLVMIR: %{{[0-9]+}} = bitcast <4 x i32> %[[r]] to <4 x float>
end subroutine vec_xor_testf32

! CHECK-LABEL: vec_xor_testf64
subroutine vec_xor_testf64(arg1, arg2)
  vector(real(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <2 x double> %[[arg1]] to <2 x i64>
! LLVMIR: %[[bc2:.*]] = bitcast <2 x double> %[[arg2]] to <2 x i64>
! LLVMIR: %[[r:.*]] = xor <2 x i64> %[[bc1]], %[[bc2]]
! LLVMIR: %{{[0-9]+}} = bitcast <2 x i64> %[[r]] to <2 x double>
end subroutine vec_xor_testf64

