! Test lowering of allocatable components
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module acomp
  implicit none
  type t
    real :: x
    integer :: i
  end type
  interface
    subroutine takes_real_scalar(x)
      real :: x
    end subroutine
    subroutine takes_char_scalar(x)
      character(*) :: x
    end subroutine
    subroutine takes_derived_scalar(x)
      import t
      type(t) :: x
    end subroutine
    subroutine takes_real_array(x)
      real :: x(:)
    end subroutine
    subroutine takes_char_array(x)
      character(*) :: x(:)
    end subroutine
    subroutine takes_derived_array(x)
      import t
      type(t) :: x(:)
    end subroutine
    subroutine takes_real_scalar_pointer(x)
      real, allocatable :: x
    end subroutine
    subroutine takes_real_array_pointer(x)
      real, allocatable :: x(:)
    end subroutine
    subroutine takes_logical(x)
      logical :: x
    end subroutine
  end interface

  type real_a0
    real, allocatable :: p
  end type
  type real_a1
    real, allocatable :: p(:)
  end type
  type cst_char_a0
    character(10), allocatable :: p
  end type
  type cst_char_a1
    character(10), allocatable :: p(:)
  end type
  type def_char_a0
    character(:), allocatable :: p
  end type
  type def_char_a1
    character(:), allocatable :: p(:)
  end type
  type derived_a0
    type(t), allocatable :: p
  end type
  type derived_a1
    type(t), allocatable :: p(:)
  end type

  real, target :: real_target, real_array_target(100)
  character(10), target :: char_target, char_array_target(100)

contains

! -----------------------------------------------------------------------------
!            Test allocatable component references
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPref_scalar_real_a(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>,
! CHECK-SAME: %[[arg2:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>>,
! CHECK-SAME: %[[arg3:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>)
subroutine ref_scalar_real_a(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg0]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) : (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_0%p)

  ! CHECK: %[[a0_1_coor:.*]] = fir.coordinate_of %[[arg2]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) : (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg1]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[addr]], %c7{{.*}} : (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) : (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_0%p(7))

  ! CHECK: %[[a1_1_coor:.*]] = fir.coordinate_of %[[arg3]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[addr]], %c7{{.*}} : (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) : (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_1(5)%p(7))
end subroutine


! CHECK-LABEL: func @_QMacompPref_array_real_a(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>
subroutine ref_array_real_a(a1_0, a1_1)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[fld_coor:.*]] = fir.coordinate_of %[[arg0]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[box:.*]] = fir.load %[[fld_coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %c20{{.*}}, %c50{{.*}}, %c2{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) [%[[slice]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QPtakes_real_array(%[[embox]]) : (!fir.box<!fir.array<?xf32>>) -> ()
  call takes_real_array(a1_0%p(20:50:2))


  ! CHECK: %[[a1_1_coor:.*]] = fir.coordinate_of %[[arg1]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[fld_coor:.*]] = fir.coordinate_of %[[a1_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[box:.*]] = fir.load %[[fld_coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %c20{{.*}}, %c50{{.*}}, %c2{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[embox:.*]] = fir.embox %[[addr]](%[[shape]]) [%[[slice]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QPtakes_real_array(%[[embox]]) : (!fir.box<!fir.array<?xf32>>) -> ()
  call takes_real_array(a1_1(5)%p(20:50:2))
end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_cst_char_a
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine ref_scalar_cst_char_a(a0_0, a1_0, a0_1, a1_1)
  type(cst_char_a0) :: a0_0, a0_1(100)
  type(cst_char_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[base]], %c7{{.*}}
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[base]], %c7{{.*}}
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_def_char_a
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine ref_scalar_def_char_a(a0_0, a1_0, a0_1, a1_1)
  type(def_char_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.coordinate_of %[[base]], %c7{{.*}}
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.coordinate_of %[[base]], %c7{{.*}}
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_derived
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine ref_scalar_derived(a0_0, a1_0, a0_1, a1_1)
  type(derived_a0) :: a0_0, a0_1(100)
  type(derived_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(a0_0%p%x)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[box]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(a0_1(5)%p%x)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %c7{{.*}}
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[elem]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(a1_0%p(7)%x)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %c7{{.*}}
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[elem]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(a1_1(5)%p(7)%x)

end subroutine

! -----------------------------------------------------------------------------
!            Test passing allocatable component references as allocatables
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPpass_real_a
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine pass_real_a(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(a1_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(a1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test usage in intrinsics where pointer aspect matters
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPallocated_p
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine allocated_p(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a0_0%p))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a0_1(5)%p))

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a1_0%p))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a1_1(5)%p))
end subroutine

! -----------------------------------------------------------------------------
!            Test allocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPallocate_real
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine allocate_real(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMacompPallocate_cst_char
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine allocate_cst_char(a0_0, a1_0, a0_1, a1_1)
  type(cst_char_a0) :: a0_0, a0_1(100)
  type(cst_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMacompPallocate_def_char
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine allocate_def_char(a0_0, a1_0, a0_1, a1_1)
  type(def_char_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a1_0%p(100))

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a1_1(5)%p(100))
end subroutine

! -----------------------------------------------------------------------------
!            Test deallocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPdeallocate_real
! CHECK-SAME: (%[[a0_0:.*]]: {{.*}}, %[[a1_0:.*]]: {{.*}}, %[[a0_1:.*]]: {{.*}}, %[[a1_1:.*]]: {{.*}})
subroutine deallocate_real(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a1_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test a recursive derived type reference
! -----------------------------------------------------------------------------

! CHECK: func @_QMacompPtest_recursive
! CHECK-SAME: (%[[x:.*]]: {{.*}})
subroutine test_recursive(x)
  type t
    integer :: i
    type(t), allocatable :: next
  end type
  type(t) :: x

  ! CHECK: %[[fldNext1:.*]] = fir.field_index next
  ! CHECK: %[[next1:.*]] = fir.coordinate_of %[[x]], %[[fldNext1]]
  ! CHECK: %[[nextBox1:.*]] = fir.load %[[next1]]
  ! CHECK: %[[fldNext2:.*]] = fir.field_index next
  ! CHECK: %[[next2:.*]] = fir.coordinate_of %[[nextBox1]], %[[fldNext2]]
  ! CHECK: %[[nextBox2:.*]] = fir.load %[[next2]]
  ! CHECK: %[[fldNext3:.*]] = fir.field_index next
  ! CHECK: %[[next3:.*]] = fir.coordinate_of %[[nextBox2]], %[[fldNext3]]
  ! CHECK: %[[nextBox3:.*]] = fir.load %[[next3]]
  ! CHECK: %[[fldi:.*]] = fir.field_index i
  ! CHECK: %[[i:.*]] = fir.coordinate_of %[[nextBox3]], %[[fldi]]
  ! CHECK: %[[nextBox3:.*]] = fir.load %[[i]] : !fir.ref<i32>
  print *, x%next%next%next%i
end subroutine

end module
