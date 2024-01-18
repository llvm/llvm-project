! RUN: bbc %s -o - | tco | FileCheck %s
! RUN: %flang -emit-llvm -S -mmlir -disable-external-name-interop %s -o - | FileCheck %s

! CHECK: @__BLNK__ = common global [8 x i8] zeroinitializer
! CHECK: @rien_ = common global [1 x i8] zeroinitializer
! CHECK: @with_empty_equiv_ = common global [8 x i8] zeroinitializer
! CHECK: @x_ = global { float, float } { float 1.0{{.*}}, float 2.0{{.*}} }
! CHECK: @y_ = common global [12 x i8] zeroinitializer
! CHECK: @z_ = global { i32, [4 x i8], float } { i32 42, [4 x i8] zeroinitializer, float 3.000000e+00 }

! CHECK-LABEL: _QPs0
subroutine s0
  common // a0, b0

  ! CHECK: call void @_QPs(ptr @__BLNK__, ptr getelementptr (i8, ptr @__BLNK__, i64 4))
  call s(a0, b0)
end subroutine s0

! CHECK-LABEL: _QPs1
subroutine s1
  common /x/ a1, b1
  data a1 /1.0/, b1 /2.0/

  ! CHECK: call void @_QPs(ptr @x_, ptr getelementptr (i8, ptr @x_, i64 4))
  call s(a1, b1)
end subroutine s1

! CHECK-LABEL: _QPs2
subroutine s2
  common /y/ a2, b2, c2

  ! CHECK: call void @_QPs(ptr @y_, ptr getelementptr (i8, ptr @y_, i64 4))
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
  ! CHECK: load i32, ptr @c_in_mod_
  print *, i
  ! CHECK: load i32, ptr getelementptr (i8, ptr @c_in_mod_, i64 4)
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
