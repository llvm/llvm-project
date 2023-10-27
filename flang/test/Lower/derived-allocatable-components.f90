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
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>{{.*}}, %[[arg2:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine ref_scalar_real_a(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg0]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_0%p)

  ! CHECK: %[[a0_1_coor:.*]] = fir.coordinate_of %[[arg2]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a0_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_1(5)%p)

  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arg1]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[addr]], %[[index]] : (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_0%p(7))

  ! CHECK: %[[a1_1_coor:.*]] = fir.coordinate_of %[[arg3]], %{{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK: %[[fld:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_1_coor]], %[[fld]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[addr]], %[[index]] : (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[coor]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func @_QMacompPref_array_real_a(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_7:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_8:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_14:.*]] = fir.shape_shift %[[VAL_6]]#0, %[[VAL_6]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_15:.*]] = fir.slice %[[VAL_9]], %[[VAL_13]], %[[VAL_11]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_16:.*]] = fir.embox %[[VAL_7]](%[[VAL_14]]) {{\[}}%[[VAL_15]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[VAL_16_NEW:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[VAL_16_NEW]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         %[[VAL_17:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_19:.*]] = arith.subi %[[VAL_17]], %[[VAL_18]] : i64
! CHECK:         %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_19]] : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_21:.*]] = fir.field_index p, !fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_20]], %[[VAL_21]] : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_24]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_26:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_27:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:         %[[VAL_29:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:         %[[VAL_31:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:         %[[VAL_33:.*]] = fir.shape_shift %[[VAL_25]]#0, %[[VAL_25]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_34:.*]] = fir.slice %[[VAL_28]], %[[VAL_32]], %[[VAL_30]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_35:.*]] = fir.embox %[[VAL_26]](%[[VAL_33]]) {{\[}}%[[VAL_34]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[VAL_35_NEW:.*]] = fir.convert %[[VAL_35]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[VAL_35_NEW]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine ref_array_real_a(a1_0, a1_1)
  type(real_a1) :: a1_0, a1_1(100)
  call takes_real_array(a1_0%p(20:50:2))
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
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[base]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[base]], %[[index]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %c10{{.*}}
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
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a0_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a1_0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[base]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[c7:.*]] = fir.convert %c7{{.*}} : (i64) -> index 
  ! CHECK: %[[sub:.*]] = arith.subi %[[c7]], %[[dims]]#0 : index 
  ! CHECK: %[[mul:.*]] = arith.muli %[[len]], %[[sub]] : index 
  ! CHECK: %[[offset:.*]] = arith.addi %[[mul]], %c0{{.*}} : index
  ! CHECK: %[[cnvt:.*]] = fir.convert %[[cast]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[cnvt]], %[[offset]]
  ! CHECK: %[[cnvt:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cnvt]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK-DAG: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK-DAG: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[base]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[c7:.*]] = fir.convert %c7{{.*}} : (i64) -> index 
  ! CHECK: %[[sub:.*]] = arith.subi %[[c7]], %[[dims]]#0 : index 
  ! CHECK: %[[mul:.*]] = arith.muli %[[len]], %[[sub]] : index 
  ! CHECK: %[[offset:.*]] = arith.addi %[[mul]], %c0{{.*}} : index
  ! CHECK: %[[cnvt:.*]] = fir.convert %[[cast]]
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[cnvt]], %[[offset]]
  ! CHECK: %[[cnvt:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cnvt]], %[[len]]
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
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %[[index]]
  ! CHECK: %[[fldx:.*]] = fir.field_index x
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[elem]], %[[fldx]]
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[addr]])
  call takes_real_scalar(a1_0%p(7)%x)

  ! CHECK-DAG: %[[coor0:.*]] = fir.coordinate_of %[[a1_1]], %{{.*}}
  ! CHECK-DAG: %[[fld:.*]] = fir.field_index p
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[coor0]], %[[fld]]
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}}
  ! CHECK: %[[lb:.*]] = fir.convert %[[dims]]#0 : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c7{{.*}}, %[[lb]] : i64
  ! CHECK: %[[elem:.*]] = fir.coordinate_of %[[box]], %[[index]]
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
