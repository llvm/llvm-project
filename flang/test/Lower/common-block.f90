! RUN: bbc %s -o - | tco | FileCheck %s

! CHECK: @_QB = common global [8 x i8] zeroinitializer
! CHECK: @_QBrien = common global [1 x i8] zeroinitializer
! CHECK: @_QBwith_empty_equiv = common global [8 x i8] zeroinitializer
! CHECK: @_QBx = global { float, float } { float 1.0{{.*}}, float 2.0{{.*}} }
! CHECK: @_QBy = common global [12 x i8] zeroinitializer
! CHECK: @_QBz = global { i32, [4 x i8], float } { i32 42, [4 x i8] undef, float 3.000000e+00 }

! CHECK-LABEL: _QPs0
subroutine s0
  common // a0, b0

  ! CHECK: call void @_QPs(float* bitcast ([8 x i8]* @_QB to float*), float* bitcast (i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_QB, i32 0, i64 4) to float*))
  call s(a0, b0)
end subroutine s0

! CHECK-LABEL: _QPs1
subroutine s1
  common /x/ a1, b1
  data a1 /1.0/, b1 /2.0/

  ! CHECK: call void @_QPs(float* getelementptr inbounds ({ float, float }, { float, float }* @_QBx, i32 0, i32 0), float* bitcast (i8* getelementptr (i8, i8* bitcast ({ float, float }* @_QBx to i8*), i64 4) to float*))
  call s(a1, b1)
end subroutine s1

! CHECK-LABEL: _QPs2
subroutine s2
  common /y/ a2, b2, c2

  ! CHECK: call void @_QPs(float* bitcast ([12 x i8]* @_QBy to float*), float* bitcast (i8* getelementptr inbounds ([12 x i8], [12 x i8]* @_QBy, i32 0, i64 4) to float*))
  call s(a2, b2)
end subroutine s2

! Test that common initialized through aliases of common members are getting
! the correct initializer.
! CHECK-LABEL: _QPs3
subroutine s3
 integer :: i = 42
 real :: x
 complex :: c
 real :: glue(2)
 real :: y = 3.
 equivalence (i, x), (glue(1), c), (glue(2), y)
 ! x and c are not directly initialized, but overlapping aliases are.
 common /z/ x, c
end subroutine s3

module mod_with_common
  integer :: i, j
  common /c_in_mod/ i, j
end module
! CHECK-LABEL: _QPs4
subroutine s4
  use mod_with_common
  ! CHECK: load i32, i32* bitcast ([8 x i8]* @_QBc_in_mod to i32*)
  print *, i
  ! CHECK: load i32, i32* bitcast (i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_QBc_in_mod, i32 0, i64 4) to i32*)
  print *, j
end subroutine s4

! CHECK-LABEL: _QPs5
subroutine s5
  real r(1:0)
  common /rien/ r
end subroutine s5

! CHECK-LABEL: _QPs6
subroutine s6
  real r1(1:0), r2(1:0), x, y
  common /with_empty_equiv/ x, r1, y
  equivalence(r1, r2)
end subroutine s6
