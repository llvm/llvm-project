! Test default initialization of local and dummy variables (dynamic initialization)
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module test_dinit
  type t
    integer :: i = 42 
  end type
  type t_alloc_comp
    real, allocatable :: i(:)
  end type
  type tseq
    sequence
    integer :: i = 42 
  end type
contains

! -----------------------------------------------------------------------------
!            Test default initialization of local and dummy variables.
! -----------------------------------------------------------------------------

  ! Test local scalar is default initialized
  ! CHECK-LABEL: func @_QMtest_dinitPlocal()
  subroutine local
    ! CHECK: %[[x:.*]] = fir.alloca !fir.type<_QMtest_dinitTt{i:i32}>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTt{i:i32}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    type(t) :: x
    print *, x%i
  end subroutine 

  ! Test local array is default initialized
  ! CHECK-LABEL: func @_QMtest_dinitPlocal_array()
  subroutine local_array()
    ! CHECK: %[[x:.*]] = fir.alloca !fir.array<4x!fir.type<_QMtest_dinitTt{i:i32}>>
    ! CHECK: %[[xshape:.*]] = fir.shape %c4{{.*}} : (index) -> !fir.shape<1>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]](%[[xshape]]) : (!fir.ref<!fir.array<4x!fir.type<_QMtest_dinitTt{i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.type<_QMtest_dinitTt{i:i32}>>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    type(t) :: x(4)
    print *, x(2)%i
  end subroutine 

  ! Test allocatable component triggers default initialization of local
  ! scalars.
  ! CHECK-LABEL: func @_QMtest_dinitPlocal_alloc_comp()
  subroutine local_alloc_comp
    ! CHECK: %[[x:.*]] = fir.alloca !fir.type<_QMtest_dinitTt_alloc_comp{i:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt_alloc_comp{i:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMtest_dinitTt_alloc_comp{i:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    type(t_alloc_comp) :: x
  end subroutine 

  ! Test function results are default initialized.
  ! CHECK-LABEL: func @_QMtest_dinitPresult() -> !fir.type<_QMtest_dinitTt{i:i32}>
  function result()
    ! CHECK: %[[x:.*]] = fir.alloca !fir.type<_QMtest_dinitTt{i:i32}>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTt{i:i32}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    type(t) :: result
  end function

  ! Test intent(out) dummies are default initialized
  ! CHECK-LABEL: func @_QMtest_dinitPintent_out(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>
  subroutine intent_out(x)
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTt{i:i32}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    type(t), intent(out) :: x
  end subroutine

  ! Test that optional intent(out) are default initialized only when
  ! present.
  ! CHECK-LABEL: func @_QMtest_dinitPintent_out_optional(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>> {fir.bindc_name = "x", fir.optional})
  subroutine intent_out_optional(x)
    ! CHECK: %[[isPresent:.*]] = fir.is_present %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>) -> i1
    ! CHECK: fir.if %[[isPresent]] {
      ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QMtest_dinitTt{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTt{i:i32}>>
      ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
      ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    ! CHECK: }
    type(t), intent(out), optional :: x
  end subroutine

  ! Test local equivalences where one entity has default initialization
  ! CHECK-LABEL: func @_QMtest_dinitPlocal_eq()
  subroutine local_eq()
    type(tseq) :: x
    integer :: zi
    ! CHECK: %[[equiv:.*]] = fir.alloca !fir.array<4xi8>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[equiv]], %c0{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    ! CHECK: %[[x:.*]] = fir.convert %[[xcoor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    equivalence (x, zi)
    print *, i
  end subroutine

  ! Test local equivalences with both equivalenced entities being
  ! default initialized. Note that the standard allow default initialization
  ! to be performed several times as long as the values are the same. So
  ! far that is what lowering is doing to stay simple.
  ! CHECK-LABEL: func @_QMtest_dinitPlocal_eq2()
  subroutine local_eq2()
    type(tseq) :: x
    type(tseq) :: y
    ! CHECK: %[[equiv:.*]] = fir.alloca !fir.array<4xi8>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[equiv]], %c0{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    ! CHECK: %[[x:.*]] = fir.convert %[[xcoor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[xboxNone:.*]] = fir.convert %[[xbox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[xboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none

  
    ! CHECK: %[[ycoor:.*]] = fir.coordinate_of %[[equiv]], %c0{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    ! CHECK: %[[y:.*]] = fir.convert %[[ycoor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[ybox:.*]] = fir.embox %[[y]] : (!fir.ptr<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
    ! CHECK: %[[yboxNone:.*]] = fir.convert %[[ybox]]
    ! CHECK: fir.call @_FortranAInitialize(%[[yboxNone]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
    equivalence (x, y)
    print *, y%i
  end subroutine


! -----------------------------------------------------------------------------
!        Test for local and dummy variables that must not be initialized
! -----------------------------------------------------------------------------

  ! CHECK-LABEL: func @_QMtest_dinitPnoinit_local_alloc
  subroutine noinit_local_alloc
    ! CHECK-NOT: fir.call @_FortranAInitialize
    type(t), allocatable :: x
    ! CHECK: return
  end subroutine 

  ! CHECK-LABEL: func @_QMtest_dinitPnoinit_local_pointer
  subroutine noinit_local_pointer
    ! CHECK-NOT: fir.call @_FortranAInitialize
    type(t), pointer :: x
    ! CHECK: return
  end subroutine 

  ! CHECK-LABEL: func @_QMtest_dinitPnoinit_normal_dummy
  subroutine noinit_normal_dummy(x)
    ! CHECK-NOT: fir.call @_FortranAInitialize
    type(t) :: x
    ! CHECK: return
  end subroutine

  ! CHECK-LABEL: func @_QMtest_dinitPnoinit_intentinout_dummy
  subroutine noinit_intentinout_dummy(x)
    ! CHECK-NOT: fir.call @_FortranAInitialize
    type(t), intent(inout) :: x
    ! CHECK: return
  end subroutine 

end module

! End-to-end test for debug pruposes.
  use test_dinit
  type(t) :: at
  call local()
  call local_array()
  at%i = 66
  call intent_out(at)
  print *, at%i
  at%i = 66
  call intent_out_optional(at)
  print *, at%i
  call intent_out_optional()
  call local_eq()
  call local_eq2()
end
