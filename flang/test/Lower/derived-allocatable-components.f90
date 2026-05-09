! Test lowering of allocatable components
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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

  ! CHECK: %[[a0_0_decl:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QMacompFref_scalar_real_aEa0_0"}
  ! CHECK: %[[a0_1_decl:.*]]:2 = hlfir.declare %[[arg2]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_real_aEa0_1"}
  ! CHECK: %[[a1_0_decl:.*]]:2 = hlfir.declare %[[arg1]]{{.*}}{uniq_name = "_QMacompFref_scalar_real_aEa1_0"}
  ! CHECK: %[[a1_1_decl:.*]]:2 = hlfir.declare %[[arg3]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_real_aEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0_decl]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1_decl]]#0 (%{{.*}})  : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>>, index) -> !fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMacompTreal_a0{p:!fir.box<!fir.heap<f32>>}>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[cast]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0_decl]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[elem]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_0%p(7))

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1_decl]]#0 (%{{.*}})  : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[elem]]) {{.*}}: (!fir.ref<f32>) -> ()
  call takes_real_scalar(a1_1(5)%p(7))
end subroutine

! CHECK-LABEL: func @_QMacompPref_array_real_a(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}) {
! CHECK:         %[[a1_0_decl:.*]]:2 = hlfir.declare %[[VAL_0]]{{.*}}{uniq_name = "_QMacompFref_array_real_aEa1_0"}
! CHECK:         %[[a1_1_decl:.*]]:2 = hlfir.declare %[[VAL_1]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_array_real_aEa1_1"}
! CHECK:         %[[coor:.*]] = hlfir.designate %[[a1_0_decl]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
! CHECK:         %[[box:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[c20:.*]] = arith.constant 20 : index
! CHECK:         %[[c50:.*]] = arith.constant 50 : index
! CHECK:         %[[c2:.*]] = arith.constant 2 : index
! CHECK:         %[[c16:.*]] = arith.constant 16 : index
! CHECK:         %[[shape:.*]] = fir.shape %[[c16]] : (index) -> !fir.shape<1>
! CHECK:         %[[slice:.*]] = hlfir.designate %[[box]] (%[[c20]]:%[[c50]]:%[[c2]])  shape %[[shape]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[cast:.*]] = fir.convert %[[slice]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[cast]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         %[[elem:.*]] = hlfir.designate %[[a1_1_decl]]#0 (%{{.*}})  : (!fir.ref<!fir.array<100x!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, index) -> !fir.ref<!fir.type<_QMacompTreal_a1{p:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:         %[[coor2:.*]] = hlfir.designate %[[elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
! CHECK:         %[[box2:.*]] = fir.load %[[coor2]]
! CHECK:         %[[slice2:.*]] = hlfir.designate %[[box2]] (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<16xf32>>
! CHECK:         %[[cast2:.*]] = fir.convert %[[slice2]] : (!fir.box<!fir.array<16xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPtakes_real_array(%[[cast2]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine ref_array_real_a(a1_0, a1_1)
  type(real_a1) :: a1_0, a1_1(100)
  call takes_real_array(a1_0%p(20:50:2))
  call takes_real_array(a1_1(5)%p(20:50:2))
end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_cst_char_a
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine ref_scalar_cst_char_a(a0_0, a1_0, a0_1, a1_1)
  type(cst_char_a0) :: a0_0, a0_1(100)
  type(cst_char_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_cst_char_aEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_cst_char_aEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_cst_char_aEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_cst_char_aEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[cast]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[elem]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  typeparams %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[elem]], %c10{{.*}}
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_def_char_a
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine ref_scalar_def_char_a(a0_0, a1_0, a0_1, a1_1)
  type(def_char_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_def_char_aEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_def_char_aEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_def_char_aEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_def_char_aEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[box2:.*]] = fir.load %[[coor]]
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box2]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[box2:.*]] = fir.load %[[coor]]
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box2]]
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[addr]], %[[len]]
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[boxchar]])
  call takes_char_scalar(a0_1(5)%p)


  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  typeparams %[[len]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[elem]])
  call takes_char_scalar(a1_0%p(7))


  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  typeparams %[[len]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPtakes_char_scalar(%[[elem]])
  call takes_char_scalar(a1_1(5)%p(7))

end subroutine

! CHECK-LABEL: func @_QMacompPref_scalar_derived
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine ref_scalar_derived(a0_0, a1_0, a0_1, a1_1)
  type(derived_a0) :: a0_0, a0_1(100)
  type(derived_a1) :: a1_0, a1_1(100)

  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_derivedEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_derivedEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFref_scalar_derivedEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFref_scalar_derivedEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[base]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[xcoor]])
  call takes_real_scalar(a0_0%p%x)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[base:.*]] = fir.box_addr %[[box]]
  ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[base]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[xcoor]])
  call takes_real_scalar(a0_1(5)%p%x)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMacompTt{x:f32,i:i32}>>>>, index) -> !fir.ref<!fir.type<_QMacompTt{x:f32,i:i32}>>
  ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[elem]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[xcoor]])
  call takes_real_scalar(a1_0%p(7)%x)

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: %[[elem:.*]] = hlfir.designate %[[box]] (%c7{{.*}})  : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMacompTt{x:f32,i:i32}>>>>, index) -> !fir.ref<!fir.type<_QMacompTt{x:f32,i:i32}>>
  ! CHECK: %[[xcoor:.*]] = hlfir.designate %[[elem]]{"x"}
  ! CHECK: fir.call @_QPtakes_real_scalar(%[[xcoor]])
  call takes_real_scalar(a1_1(5)%p(7)%x)

end subroutine

! -----------------------------------------------------------------------------
!            Test passing allocatable component references as allocatables
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPpass_real_a
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine pass_real_a(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFpass_real_aEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFpass_real_aEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFpass_real_aEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFpass_real_aEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.call @_QPtakes_real_scalar_pointer(%[[coor]])
  call takes_real_scalar_pointer(a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(a1_0%p)

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.call @_QPtakes_real_array_pointer(%[[coor]])
  call takes_real_array_pointer(a1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test usage in intrinsics where pointer aspect matters
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPallocated_p
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine allocated_p(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFallocated_pEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocated_pEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFallocated_pEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocated_pEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a0_0%p))

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a0_1(5)%p))

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a1_0%p))

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[box:.*]] = fir.load %[[coor]]
  ! CHECK: fir.box_addr %[[box]]
  call takes_logical(allocated(a1_1(5)%p))
end subroutine

! -----------------------------------------------------------------------------
!            Test allocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPallocate_real
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine allocate_real(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_realEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_realEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_realEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_realEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_0%p(100))

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMacompPallocate_cst_char
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine allocate_cst_char(a0_0, a1_0, a0_1, a1_1)
  type(cst_char_a0) :: a0_0, a0_1(100)
  type(cst_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_cst_charEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_cst_charEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_cst_charEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_cst_charEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_0%p(100))

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}} typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(a1_1(5)%p(100))
end subroutine

! CHECK-LABEL: func @_QMacompPallocate_def_char
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine allocate_def_char(a0_0, a1_0, a0_1, a1_1)
  type(def_char_a0) :: a0_0, a0_1(100)
  type(def_char_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_def_charEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_def_charEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFallocate_def_charEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFallocate_def_charEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a1_0%p(100))

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  allocate(character(18)::a1_1(5)%p(100))
end subroutine

! -----------------------------------------------------------------------------
!            Test deallocation
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMacompPdeallocate_real
! CHECK-SAME: (%[[a0_0_arg:.*]]: {{.*}}, %[[a1_0_arg:.*]]: {{.*}}, %[[a0_1_arg:.*]]: {{.*}}, %[[a1_1_arg:.*]]: {{.*}})
subroutine deallocate_real(a0_0, a1_0, a0_1, a1_1)
  type(real_a0) :: a0_0, a0_1(100)
  type(real_a1) :: a1_0, a1_1(100)
  ! CHECK: %[[a0_0:.*]]:2 = hlfir.declare %[[a0_0_arg]]{{.*}}{uniq_name = "_QMacompFdeallocate_realEa0_0"}
  ! CHECK: %[[a0_1:.*]]:2 = hlfir.declare %[[a0_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFdeallocate_realEa0_1"}
  ! CHECK: %[[a1_0:.*]]:2 = hlfir.declare %[[a1_0_arg]]{{.*}}{uniq_name = "_QMacompFdeallocate_realEa1_0"}
  ! CHECK: %[[a1_1:.*]]:2 = hlfir.declare %[[a1_1_arg]](%{{.*}}){{.*}}{uniq_name = "_QMacompFdeallocate_realEa1_1"}

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a0_0%p)

  ! CHECK: %[[a0_1_elem:.*]] = hlfir.designate %[[a0_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a0_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a0_1(5)%p)

  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_0]]#0{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a1_0%p)

  ! CHECK: %[[a1_1_elem:.*]] = hlfir.designate %[[a1_1]]#0 (%{{.*}})
  ! CHECK: %[[coor:.*]] = hlfir.designate %[[a1_1_elem]]{"p"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: fir.store {{.*}} to %[[coor]]
  deallocate(a1_1(5)%p)
end subroutine

! -----------------------------------------------------------------------------
!            Test a recursive derived type reference
! -----------------------------------------------------------------------------

! CHECK: func @_QMacompPtest_recursive
! CHECK-SAME: (%[[xarg:.*]]: {{.*}})
subroutine test_recursive(x)
  type t
    integer :: i
    type(t), allocatable :: next
  end type
  type(t) :: x

  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xarg]]{{.*}}{uniq_name = "_QMacompFtest_recursiveEx"}
  ! CHECK: %[[next1:.*]] = hlfir.designate %[[x]]#0{"next"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[nextBox1:.*]] = fir.load %[[next1]]
  ! CHECK: %[[next1addr:.*]] = fir.box_addr %[[nextBox1]]
  ! CHECK: %[[next2:.*]] = hlfir.designate %[[next1addr]]{"next"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[nextBox2:.*]] = fir.load %[[next2]]
  ! CHECK: %[[next2addr:.*]] = fir.box_addr %[[nextBox2]]
  ! CHECK: %[[next3:.*]] = hlfir.designate %[[next2addr]]{"next"}{{.*}}{fortran_attrs = #fir.var_attrs<allocatable>}
  ! CHECK: %[[nextBox3:.*]] = fir.load %[[next3]]
  ! CHECK: %[[next3addr:.*]] = fir.box_addr %[[nextBox3]]
  ! CHECK: %[[i:.*]] = hlfir.designate %[[next3addr]]{"i"}
  ! CHECK: %[[ival:.*]] = fir.load %[[i]] : !fir.ref<i32>
  print *, x%next%next%next%i
end subroutine

end module
