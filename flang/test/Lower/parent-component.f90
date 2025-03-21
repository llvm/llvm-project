! Test different ways of passing the parent component of an extended
! derived-type to a subroutine or the runtime.

! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

program parent_comp
  type p
    integer :: a
  end type

  type, extends(p) :: c
    integer :: b
  end type

  type z
    integer :: k
    type(c) :: c
  end type

  type(c) :: t(2) = [ c(11, 21), c(12, 22) ]
  call init_with_slice()
  call init_no_slice()
  call init_allocatable()
  call init_scalar()
  call init_assumed(t)
contains

  subroutine print_scalar(a)
    type(p), intent(in) :: a
    print*, a
  end subroutine
  ! CHECK-LABEL: func.func private @_QFPprint_scalar(%{{.*}}: !fir.ref<!fir.type<_QFTp{a:i32}>> {fir.bindc_name = "a"})

  subroutine print_p(a)
    type(p), intent(in) :: a(2)
    print*, a
  end subroutine
  ! CHECK-LABEL: func.func private @_QFPprint_p(%{{.*}}: !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>> {fir.bindc_name = "a"})

  subroutine init_with_slice()
    type(c) :: y(2) = [ c(11, 21), c(12, 22) ]
    call print_p(y(:)%p)
    print*,y(:)%p
  end subroutine
  ! CHECK-LABEL: func.func private @_QFPinit_with_slice()
  ! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>
  ! CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFFinit_with_sliceEy) : !fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>
  ! CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  ! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = "_QFFinit_with_sliceEy"} : (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>)
  ! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
  ! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_5]]:%[[VAL_2]]:%[[VAL_6]])  shape %[[VAL_8]] : (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>
  ! CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_9]]{"p"}   shape %[[VAL_8]] : (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           %[[VAL_11:.*]]:2 = hlfir.copy_in %[[VAL_10]] to %[[VAL_0]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>>) -> (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>, i1)
  ! CHECK:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]]#0 : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           fir.call @_QFPprint_p(%[[VAL_12]]) fastmath<contract> : (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()
  ! CHECK:           hlfir.copy_out %[[VAL_0]], %[[VAL_11]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>>, i1) -> ()

  subroutine init_no_slice()
    type(c) :: y(2) = [ c(11, 21), c(12, 22) ]
    call print_p(y%p)
    print*,y%p
  end subroutine
  ! CHECK-LABEL: func.func private @_QFPinit_no_slice()
  ! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>
  ! CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFFinit_no_sliceEy) : !fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>
  ! CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  ! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = "_QFFinit_no_sliceEy"} : (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>)
  ! CHECK:           %[[VAL_5:.*]] = hlfir.designate %[[VAL_4]]#0{"p"}   shape %[[VAL_3]] : (!fir.ref<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           %[[VAL_6:.*]]:2 = hlfir.copy_in %[[VAL_5]] to %[[VAL_0]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>>) -> (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>, i1)
  ! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]]#0 : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           fir.call @_QFPprint_p(%[[VAL_7]]) fastmath<contract> : (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()
  ! CHECK:           hlfir.copy_out %[[VAL_0]], %[[VAL_6]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>>>, i1) -> ()

  subroutine init_allocatable()
    type(c), allocatable :: y(:)
    allocate(y(2))
    y(1) = c(11, 21)
    y(2) = c(12, 22)
    call print_p(y%p)
    print*,y%p
  end subroutine

  ! CHECK-LABEL: func.func private @_QFPinit_allocatable()
  ! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}_QFFinit_allocatableEy"
  ! CHECK:           hlfir.assign
  ! CHECK:           hlfir.assign
  ! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>>>
  ! CHECK:           %[[VAL_31:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_30]], %[[VAL_31]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_33:.*]] = fir.shape %[[VAL_32]]#1 : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_34:.*]] = hlfir.designate %[[VAL_30]]{"p"}   shape %[[VAL_33]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           %[[VAL_35:.*]]:2 = hlfir.copy_in %[[VAL_34]] to %[[VAL_0:.*]] : (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTp{a:i32}>>>>>) -> (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>, i1)
  ! CHECK:           %[[VAL_36:.*]] = fir.box_addr %[[VAL_35]]#0 : (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<?x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK:           fir.call @_QFPprint_p(%[[VAL_37]]) fastmath<contract> : (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()
  ! CHECK:           hlfir.copy_out %[[VAL_0]], %[[VAL_35]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTp{a:i32}>>>>>, i1) -> ()

  subroutine init_scalar()
    type(c) :: s = c(11, 21)
    call print_scalar(s%p)
    print*,s%p
  end subroutine

  ! CHECK-LABEL: func.func private @_QFPinit_scalar()
  ! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFFinit_scalarEs) : !fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>
  ! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFFinit_scalarEs"} : (!fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>) -> (!fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>, !fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>)
  ! CHECK:           %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"p"}   : (!fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>) -> !fir.ref<!fir.type<_QFTp{a:i32}>>
  ! CHECK:           fir.call @_QFPprint_scalar(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.type<_QFTp{a:i32}>>) -> ()

  subroutine init_assumed(y)
    type(c) :: y(:)
    call print_p(y%p)
    print*,y%p
  end subroutine

  ! CHECK-LABEL: func.func private @_QFPinit_assumed(
  ! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}"_QFFinit_assumedEy"
  ! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_7:.*]] = hlfir.designate %[[VAL_3]]#0{"p"}   shape %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>

  subroutine init_existing_field()
    type(z) :: y(2)
    call print_p(y%c%p)
  end subroutine

  ! CHECK-LABEL: func.func private @_QFPinit_existing_field
  ! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}"_QFFinit_existing_fieldEy"
  ! CHECK:           %[[VAL_5:.*]] = hlfir.designate %[[VAL_4]]#0{"c"}   shape %[[VAL_3]] : (!fir.ref<!fir.array<2x!fir.type<_QFTz{k:i32,c:!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>
  ! CHECK:           %[[VAL_6:.*]] = hlfir.designate %[[VAL_5]]{"p"}   shape %[[VAL_3]] : (!fir.box<!fir.array<2x!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>

  subroutine parent_comp_lhs()
    type(c) :: a
    type(p) :: b

    a%p = B
  end subroutine

! CHECK-LABEL: func.func private @_QFPparent_comp_lhs
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}}"_QFFparent_comp_lhsEa"
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}"_QFFparent_comp_lhsEb"
! CHECK:           %[[VAL_4:.*]] = hlfir.designate %[[VAL_1]]#0{"p"}   : (!fir.ref<!fir.type<_QFTc{p:!fir.type<_QFTp{a:i32}>,b:i32}>>) -> !fir.ref<!fir.type<_QFTp{a:i32}>>
! CHECK:           hlfir.assign %[[VAL_3]]#0 to %[[VAL_4]] : !fir.ref<!fir.type<_QFTp{a:i32}>>, !fir.ref<!fir.type<_QFTp{a:i32}>>

end
