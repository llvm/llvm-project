! RUN: bbc -emit-hlfir -mlir-print-debuginfo -o - %s | FileCheck %s

! Test that the code of the specification part gets line 0 while the
! declarations keep the line of the variable.

subroutine adjustable(a, n)
! CHECK-LABEL: func.func @_QPadjustable(
  integer :: n
! CHECK: hlfir.declare %arg1 {{.*}}uniq_name("_QFadjustableEn"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:14)
! CHECK: fir.load %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
! CHECK: arith.muli %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
! CHECK: arith.select %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
! CHECK: fir.shape_shift %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  integer :: a(n:2*n)
! CHECK: hlfir.declare %arg0({{.*}}uniq_name("_QFadjustableEa"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:14)
  a = 1
! CHECK: hlfir.assign %{{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:3)
end subroutine

subroutine automatic(n)
! CHECK-LABEL: func.func @_QPautomatic(
  integer :: n
! CHECK: hlfir.declare %arg0 {{.*}}uniq_name("_QFautomaticEn"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:14)
! CHECK: fir.alloca !fir.array<?xf32>{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  real :: work(n)
! CHECK: hlfir.declare %{{.*}}uniq_name("_QFautomaticEwork"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:11)
  work = 1.0
! CHECK: hlfir.assign %{{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:3)
end subroutine

subroutine assumed_len(s)
! CHECK-LABEL: func.func @_QPassumed_len(
! CHECK: fir.unboxchar %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  character(*) :: s
! CHECK: hlfir.declare %{{.*}}uniq_name("_QFassumed_lenEs"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:19)
  s = 'x'
! CHECK: hlfir.assign %{{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:3)
end subroutine

subroutine automatic_char(n)
! CHECK-LABEL: func.func @_QPautomatic_char(
  integer :: n
! CHECK: fir.alloca !fir.char<1,?>{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  character(len=n) :: buf
! CHECK: hlfir.declare %{{.*}}uniq_name("_QFautomatic_charEbuf"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:23)
  buf = 'y'
! CHECK: hlfir.assign %{{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:3)
end subroutine

! Host association tuple.
subroutine host(n)
! CHECK-LABEL: func.func @_QPhost(
  integer :: n
! CHECK: fir.alloca !fir.array<?xi32>{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  integer :: a(n)
! CHECK: hlfir.declare %{{.*}}uniq_name("_QFhostEa"){{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:14)
! CHECK: fir.alloca tuple<{{.*}}> loc("{{.*}}specification-part-loc.f90":0:0)
! CHECK: fir.coordinate_of %{{.*}} loc("{{.*}}specification-part-loc.f90":0:0)
  call inner
! CHECK: fir.call @_QFhostPinner{{.*}} loc("{{.*}}specification-part-loc.f90":[[# @LINE-1]]:3)
contains
  subroutine inner
    a(1) = 1
  end subroutine
end subroutine
