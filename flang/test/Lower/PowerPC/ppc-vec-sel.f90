! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_sel
!----------------------

! CHECK-LABEL: vec_sel_testi1
subroutine vec_sel_testi1(arg1, arg2, arg3)
  vector(integer(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR:  %[[comp:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR:  %[[and1:.*]] = and <16 x i8> %[[arg1]], %[[comp]]
! LLVMIR:  %[[and2:.*]] = and <16 x i8> %[[arg2]], %[[arg3]]
! LLVMIR:  %{{[0-9]+}} = or <16 x i8> %[[and1]], %[[and2]]
end subroutine vec_sel_testi1

! CHECK-LABEL: vec_sel_testi2
subroutine vec_sel_testi2(arg1, arg2, arg3)
  vector(integer(2)) :: arg1, arg2, r
  vector(unsigned(2)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <8 x i16> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <8 x i16> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <8 x i16> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <8 x i16>
end subroutine vec_sel_testi2

! CHECK-LABEL: vec_sel_testi4
subroutine vec_sel_testi4(arg1, arg2, arg3)
  vector(integer(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <4 x i32> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <4 x i32> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x i32>
end subroutine vec_sel_testi4

! CHECK-LABEL: vec_sel_testi8
subroutine vec_sel_testi8(arg1, arg2, arg3)
  vector(integer(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <2 x i64> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <2 x i64> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x i64>
end subroutine vec_sel_testi8

! CHECK-LABEL: vec_sel_testu1
subroutine vec_sel_testu1(arg1, arg2, arg3)
  vector(unsigned(1)) :: arg1, arg2, r
  vector(unsigned(1)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! LLVMIR:  %[[comp:.*]] = xor <16 x i8> %[[arg3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR:  %[[and1:.*]] = and <16 x i8> %[[arg1]], %[[comp]]
! LLVMIR:  %[[and2:.*]] = and <16 x i8> %[[arg2]], %[[arg3]]
! LLVMIR:  %{{[0-9]+}} = or <16 x i8> %[[and1]], %[[and2]]
end subroutine vec_sel_testu1

! CHECK-LABEL: vec_sel_testu2
subroutine vec_sel_testu2(arg1, arg2, arg3)
  vector(unsigned(2)) :: arg1, arg2, r
  vector(unsigned(2)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <8 x i16> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <8 x i16> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <8 x i16> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <8 x i16>
end subroutine vec_sel_testu2

! CHECK-LABEL: vec_sel_testu4
subroutine vec_sel_testu4(arg1, arg2, arg3)
  vector(unsigned(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <4 x i32> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <4 x i32> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x i32>
end subroutine vec_sel_testu4

! CHECK-LABEL: vec_sel_testu8
subroutine vec_sel_testu8(arg1, arg2, arg3)
  vector(unsigned(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)
  

! LLVMIR: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <2 x i64> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <2 x i64> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x i64>
end subroutine vec_sel_testu8

! CHECK-LABEL: vec_sel_testr4
subroutine vec_sel_testr4(arg1, arg2, arg3)
  vector(real(4)) :: arg1, arg2, r
  vector(unsigned(4)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <4 x float> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <4 x float> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <4 x i32> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <4 x float>
end subroutine vec_sel_testr4

! CHECK-LABEL: vec_sel_testr8
subroutine vec_sel_testr8(arg1, arg2, arg3)
  vector(real(8)) :: arg1, arg2, r
  vector(unsigned(8)) :: arg3
  r = vec_sel(arg1, arg2, arg3)

! LLVMIR: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! LLVMIR: %[[arg3:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! LLVMIR: %[[bc1:.*]] = bitcast <2 x double> %5 to <16 x i8>
! LLVMIR: %[[bc2:.*]] = bitcast <2 x double> %6 to <16 x i8>
! LLVMIR: %[[bc3:.*]] = bitcast <2 x i64> %7 to <16 x i8>
! LLVMIR: %[[comp:.*]] = xor <16 x i8> %[[bc3]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
! LLVMIR: %[[and1:.*]] = and <16 x i8> %[[bc1]], %[[comp]]
! LLVMIR: %[[and2:.*]] = and <16 x i8> %[[bc2]], %[[bc3]]
! LLVMIR: %[[or:.*]] = or <16 x i8> %[[and1]], %[[and2]]
! LLVMIR: %{{[0-9]+}} = bitcast <16 x i8> %[[or]] to <2 x double>
end subroutine vec_sel_testr8
