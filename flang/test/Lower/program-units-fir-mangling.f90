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
  ! CHECK-LABEL: func private @_QFfoo2Psub() {{.*}} {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func private @_QFfoo2Pfoo() {{.*}} {
  subroutine foo()
  ! CHECK: }
  end subroutine
end function

! CHECK-LABEL: func @_QPsub2()
subroutine sUb2()
contains
  ! CHECK-LABEL: func private @_QFsub2Psub() {{.*}} {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func private @_QFsub2Pfoo() {{.*}} {
  subroutine Foo()
  ! CHECK: }
  end subroutine
end subroutine

module testMod2
contains
  ! CHECK-LABEL: func @_QMtestmod2Psub()
  subroutine sub()
  contains
    ! CHECK-LABEL: func private @_QMtestmod2FsubPsubsub() {{.*}} {
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
    ! CHECK-LABEL: func private @_QMcolor_pointsScolor_points_aSimplFfooPbar() {{.*}} {
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

! CHECK-LABEL: func @_QPshould_not_collide() {
subroutine should_not_collide()
! CHECK: }
end subroutine

! CHECK-LABEL: func @_QQmain() attributes {fir.bindc_name = "test"} {
program test
! CHECK: }
contains
! CHECK-LABEL: func private @_QFPshould_not_collide() {{.*}} {
subroutine should_not_collide()
! CHECK: }
end subroutine
end program

! CHECK-LABEL: func @omp_get_num_threads() -> f32 attributes {fir.bindc_name = "omp_get_num_threads", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
function omp_get_num_threads() bind(c)
! CHECK: }
end function

! CHECK-LABEL: func @get_threads() -> f32 attributes {fir.bindc_name = "get_threads", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
function omp_get_num_threads_1() bind(c, name ="get_threads")
! CHECK: }
end function

! CHECK-LABEL: func @bEtA() -> f32 attributes {fir.bindc_name = "bEtA", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
function alpha() bind(c, name =" bEtA ")
! CHECK: }
end function

! CHECK-LABEL: func @bc1() attributes {fir.bindc_name = "bc1", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
subroutine bind_c_s() Bind(C,Name='bc1')
  ! CHECK: return
end subroutine bind_c_s

! CHECK-LABEL: func @_QPbind_c_s() {
subroutine bind_c_s()
  ! CHECK: fir.call @_QPbind_c_q() {{.*}}: () -> ()
  ! CHECK: return
  call bind_c_q
end

! CHECK-LABEL: func @_QPbind_c_q() {
subroutine bind_c_q()
  interface
    subroutine bind_c_s() Bind(C, name='bc1')
    end
  end interface
  ! CHECK: fir.call @bc1() {{.*}}: () -> ()
  ! CHECK: return
  call bind_c_s
end

! Test that BIND(C) label is taken into account for ENTRY symbols.
! CHECK-LABEL: func @_QPsub_with_entries() {
subroutine sub_with_entries
! CHECK-LABEL: func @bar() attributes {fir.bindc_name = "bar", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
 entry some_entry() bind(c, name="bar")
! CHECK-LABEL: func @_QPnormal_entry() {
 entry normal_entry()
! CHECK-LABEL: func @some_other_entry() attributes {fir.bindc_name = "some_other_entry", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
 entry some_other_entry() bind(c)
end subroutine

! Test that semantics constructs binding labels with local name resolution
module testMod3
  character*(*), parameter :: foo = "bad!!"
  character*(*), parameter :: ok = "ok"
  interface
    real function f1() bind(c,name=ok//'1')
      import ok
    end function
    subroutine s1() bind(c,name=ok//'2')
      import ok
    end subroutine
  end interface
 contains
! CHECK-LABEL: func @ok3() -> f32 attributes {fir.bindc_name = "ok3", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  real function f2() bind(c,name=foo//'3')
    character*(*), parameter :: foo = ok
! CHECK: fir.call @ok1() {{.*}}: () -> f32
! CHECK-LABEL: func @ok4() -> f32 attributes {fir.bindc_name = "ok4", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
    entry f3() bind(c,name=foo//'4')
! CHECK: fir.call @ok1() {{.*}}: () -> f32
    f2 = f1()
  end function
! CHECK-LABEL: func @ok5() attributes {fir.bindc_name = "ok5", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  subroutine s2() bind(c,name=foo//'5')
    character*(*), parameter :: foo = ok
! CHECK: fir.call @ok2() {{.*}}: () -> ()
! CHECK-LABEL: func @ok6() attributes {fir.bindc_name = "ok6", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
    entry s3() bind(c,name=foo//'6')
! CHECK: fir.call @ok2() {{.*}}: () -> ()
    continue ! force end of specification part
! CHECK-LABEL: func @ok7() attributes {fir.bindc_name = "ok7", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
    entry s4() bind(c,name=foo//'7')
! CHECK: fir.call @ok2() {{.*}}: () -> ()
    call s1
  end subroutine
end module


! CHECK-LABEL: func @_QPnest1
subroutine nest1
  ! CHECK:   fir.call @_QFnest1Pinner()
  call inner
contains
  ! CHECK-LABEL: func private @_QFnest1Pinner
  subroutine inner
    ! CHECK:   %[[V_0:[0-9]+]] = fir.address_of(@_QFnest1FinnerEkk) : !fir.ref<i32>
    integer, save :: kk = 1
    print*, 'qq:inner', kk
  end
end

! CHECK-LABEL: func @_QPnest2
subroutine nest2
  ! CHECK:   fir.call @_QFnest2Pinner()
  call inner
contains
  ! CHECK-LABEL: func private @_QFnest2Pinner
  subroutine inner
    ! CHECK:   %[[V_0:[0-9]+]] = fir.address_of(@_QFnest2FinnerEkk) : !fir.ref<i32>
    integer, save :: kk = 77
    print*, 'ss:inner', kk
  end
end

! CHECK-LABEL: fir.global internal @_QFfooEpi : f32 {
