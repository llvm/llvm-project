! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

! CHECK-LABEL: func @_QPsub() {
subroutine sub()
! CHECK: }
end subroutine

! CHECK-LABEL: func @_QPasubroutine() {
subroutine AsUbRoUtInE()
! CHECK: }
end subroutine

! CHECK-LABEL: func @_QPfoo() -> f32 {
function foo()
  real(4) :: foo
  real :: pi = 3.14159
! CHECK: }
end function


! CHECK-LABEL: func @_QPfunctn() -> f32 {
function functn
  real, parameter :: pi = 3.14
! CHECK: }
end function


module testMod
contains
  ! CHECK-LABEL: func @_QMtestmodPsub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QMtestmodPfoo() -> f32 {
  function foo()
    real(4) :: foo
  ! CHECK: }
  end function
end module


! CHECK-LABEL: func @_QPfoo2()
function foo2()
  real(4) :: foo2
contains
  ! CHECK-LABEL: func @_QFfoo2Psub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QFfoo2Pfoo() {
  subroutine foo()
  ! CHECK: }
  end subroutine
end function

! CHECK-LABEL: func @_QPsub2()
subroutine sUb2()
contains
  ! CHECK-LABEL: func @_QFsub2Psub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QFsub2Pfoo() {
  subroutine Foo()
  ! CHECK: }
  end subroutine
end subroutine

module testMod2
contains
  ! CHECK-LABEL: func @_QMtestmod2Psub()
  subroutine sub()
  contains
    ! CHECK-LABEL: func @_QMtestmod2FsubPsubsub() {
    subroutine subSub()
    ! CHECK: }
    end subroutine
  end subroutine
end module


module color_points
  interface
    module subroutine draw()
    end subroutine
    module function erase()
      integer(4) :: erase
    end function
  end interface
end module color_points

submodule (color_points) color_points_a
contains
  ! CHECK-LABEL: func @_QMcolor_pointsScolor_points_aPsub() {
  subroutine sub
  end subroutine
  ! CHECK: }
end submodule

submodule (color_points:color_points_a) impl
contains
  ! CHECK-LABEL: func @_QMcolor_pointsScolor_points_aSimplPfoo()
  subroutine foo
    contains
    ! CHECK-LABEL: func @_QMcolor_pointsScolor_points_aSimplFfooPbar() {
    subroutine bar
    ! CHECK: }
    end subroutine
  end subroutine
  ! CHECK-LABEL: func @_QMcolor_pointsPdraw() {
  module subroutine draw()
  end subroutine
  !FIXME func @_QMcolor_pointsPerase() -> i32 {
  module procedure erase
  ! CHECK: }
  end procedure
end submodule


! CHECK-LABEL: func @_QQmain() {
program test
! CHECK: }
end program

! CHECK-LABEL: fir.global internal @_QFfooEpi : f32 {
! CHECK-LABEL: fir.global internal @_QFfunctnECpi constant : f32 {

