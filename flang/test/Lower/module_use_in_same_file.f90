! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test use of module data that is defined in this file.
! TODO: similar tests for the functions that are using the modules, but without the
! module being defined in this file. This require a front-end fix to be pushed first
! so

! Module m2 defines simple data
module m2
  real :: x
  integer :: y(100)
contains
  ! CHECK-LABEL: func @_QMm2Pfoo()
  real function foo()
    ! CHECK-DAG: fir.address_of(@_QMm2Ex) : !fir.ref<f32>
    ! CHECK-DAG: fir.address_of(@_QMm2Ey) : !fir.ref<!fir.array<100xi32>>
    foo = x + y(1)
  end function
end module
! CHECK-LABEL: func @_QPm2use()
real function m2use()
  use m2
  ! CHECK-DAG: fir.address_of(@_QMm2Ex) : !fir.ref<f32>
  ! CHECK-DAG: fir.address_of(@_QMm2Ey) : !fir.ref<!fir.array<100xi32>>
  m2use = x + y(1)
end function
! Test renaming
! CHECK-LABEL: func @_QPm2use_rename()
real function m2use_rename()
  use m2, only: renamedx => x
  ! CHECK-DAG: fir.address_of(@_QMm2Ex) : !fir.ref<f32>
  m2use_rename = renamedx
end function

! Module modEq2 defines data that is equivalenced
module modEq2
  ! Equivalence, no initialization
  real :: x1(10), x2(10), x3(10) 
  ! Equivalence with initialization
  real :: y1 = 42.
  real :: y2(10)
  equivalence (x1(1), x2(5), x3(10)), (y1, y2(5))
contains
  ! CHECK-LABEL: func @_QMmodeq2Pfoo()
  real function foo()
    ! CHECK-DAG: fir.address_of(@_QMmodeq2Ex1) : !fir.ref<!fir.array<76xi8>>
    ! CHECK-DAG: fir.address_of(@_QMmodeq2Ey1) : !fir.ref<!fir.array<10xi32>>
    foo = x2(1) + y1
  end function
end module
! CHECK-LABEL: func @_QPmodeq2use()
real function modEq2use()
  use modEq2
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ex1) : !fir.ref<!fir.array<76xi8>>
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ey1) : !fir.ref<!fir.array<10xi32>>
  modEq2use = x2(1) + y1
end function
! Test rename of used equivalence members
! CHECK-LABEL: func @_QPmodeq2use_rename()
real function modEq2use_rename()
  use modEq2, only: renamedx => x2, renamedy => y1
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ex1) : !fir.ref<!fir.array<76xi8>>
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ey1) : !fir.ref<!fir.array<10xi32>>
  modEq2use = renamedx(1) + renamedy
end function


! Module defines variable in common block
module modCommon2
  ! Module variable is in blank common
  real :: x_blank
  common // x_blank
  ! Module variable is in named common, no init
  real :: x_named1(10)
  common /named1/ x_named1
  ! Module variable is in named common, with init
  integer :: i_named2 = 42
  common /named2/ i_named2
contains
  ! CHECK-LABEL: func @_QMmodcommon2Pfoo()
  real function foo()
   ! CHECK-DAG: fir.address_of(@_QCnamed2) : !fir.ref<tuple<i32>>
   ! CHECK-DAG: fir.address_of(@_QC) : !fir.ref<!fir.array<4xi8>>
   ! CHECK-DAG: fir.address_of(@_QCnamed1) : !fir.ref<!fir.array<40xi8>>
   foo = x_blank + x_named1(5) + i_named2
  end function
end module
! CHECK-LABEL: func @_QPmodcommon2use()
real function modCommon2use()
 use modCommon2
 ! CHECK-DAG: fir.address_of(@_QCnamed2) : !fir.ref<tuple<i32>>
 ! CHECK-DAG: fir.address_of(@_QC) : !fir.ref<!fir.array<4xi8>>
 ! CHECK-DAG: fir.address_of(@_QCnamed1) : !fir.ref<!fir.array<40xi8>>
 modCommon2use = x_blank + x_named1(5) + i_named2
end function
! CHECK-LABEL: func @_QPmodcommon2use_rename()
real function modCommon2use_rename()
 use modCommon2, only : renamed0 => x_blank, renamed1 => x_named1, renamed2 => i_named2
 ! CHECK-DAG: fir.address_of(@_QCnamed2) : !fir.ref<tuple<i32>>
 ! CHECK-DAG: fir.address_of(@_QC) : !fir.ref<!fir.array<4xi8>>
 ! CHECK-DAG: fir.address_of(@_QCnamed1) : !fir.ref<!fir.array<40xi8>>
 modCommon2use_rename = renamed0 + renamed1(5) + renamed2
end function


! Test that there are no conflicts between equivalence use associated and the ones
! from the scope
real function test_no_equiv_conflicts()
  use modEq2
  ! Same equivalences as in modEq2. Test that lowering does not mixes
  ! up the equivalence based on the similar offset inside the scope.
  real :: x1l(10), x2l(10), x3l(10) 
  real :: y1l = 42.
  real :: y2l(10)
  save :: x1l, x2l, x3l, y1l, y2l
  equivalence (x1l(1), x2l(5), x3l(10)), (y1l, y2l(5))
  ! CHECK-DAG: fir.address_of(@_QFtest_no_equiv_conflictsEx1l) : !fir.ref<!fir.array<76xi8>>
  ! CHECK-DAG: fir.address_of(@_QFtest_no_equiv_conflictsEy1l) : !fir.ref<!fir.array<10xi32>>
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ex1) : !fir.ref<!fir.array<76xi8>>
  ! CHECK-DAG: fir.address_of(@_QMmodeq2Ey1) : !fir.ref<!fir.array<10xi32>>
  test_no_equiv_conflicts = x2(1) + y1 + x2l(1) + y1l
end function
