! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR
! RUN: %flang_fc1 -emit-fir   -fcoarray %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! This test checks that, when casting to a volatile type, the corank attribute is forwarded.

module m
  implicit none

  type :: t
    character(len=4) :: c(2)
  end type

contains

  subroutine display()
    class(t), volatile, allocatable, save :: cs[:]
    allocate(cs[*])
    print *, "[", cs%c(1), "][", cs%c(2), "]"
    deallocate(cs)
  end subroutine

end module

! CHECK-LABEL: func.func @_QMmPdisplay()
! CHECK:   %[[ADDR:.*]] = fir.address_of(@_QMmFdisplayEcs) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, corank:1>>
! CHECK:   %[[CAST:.*]] = fir.volatile_cast %[[ADDR]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, corank:1>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[CAST]] uniq_name("_QMmFdisplayEcs") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>)
! FIR:     %[[DECL:.*]] = fir.declare %[[CAST]] uniq_name("_QMmFdisplayEcs") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>
! FIR:     %[[UNCAST:.*]] = fir.volatile_cast %[[DECL]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>>
! CHECK:   mif.alloc_coarray %[[DECL]]{{(#0)?}} lcobounds %{{.*}} ucobounds %{{.*}} errmsg %{{.*}} uniq_name("_QMmFdisplayEcs") : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
! CHECK:   %[[LOAD:.*]] = fir.load %[[DECL]]{{(#0)?}} : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>
! HLFIR:   %{{.*}} = hlfir.designate %[[LOAD]]{"c"} {{.*}} : (!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,4>, volatile>
! FIR:     %{{.*}} = fir.coordinate_of %[[LOAD]], c : (!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>) -> !fir.ref<!fir.array<2x!fir.char<1,4>>, volatile>
! FIR:     mif.dealloc_coarray %[[DECL]] stat %{{.*}} errmsg %{{.*}} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>, !fir.ref<i32>, !fir.box<none>) -> ()

subroutine s()
  character(4), volatile, save :: c(2)[*]
  c = "x"
end subroutine

! HLFIR-LABEL: func.func @_QPs()
! HLFIR:   %[[VAL_1:.*]] = fir.address_of(@_QFsEc) : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>
! HLFIR:   %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %{{.*}} uniq_name("_QFsEc") fortran_attrs<volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>)
! HLFIR:   %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   hlfir.assign %{{.*}} to %[[VAL_6]] : !fir.ref<!fir.char<1>>, !fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>

! FIR-LABEL: func.func @_QPs()
! FIR:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<2x!fir.char<1,4>>>
! FIR-NOT:     fir.alloca !fir.box<!fir.heap<
! FIR:         %[[VAL_16:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>) -> !fir.ref<!fir.box<none>>
! FIR:         fir.call @_FortranAAssign(%[[VAL_16]], {{.*}})
subroutine s_alloc()
  character(:), volatile, allocatable :: c(:)[:]
  allocate(character(4) :: c(2)[*])
  c = "x"
end subroutine

! CHECK-LABEL: func.func @_QPs_alloc()
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, corank:1>
! CHECK:   %[[CAST:.*]] = fir.volatile_cast %[[ALLOCA]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[CAST]] uniq_name("_QFs_allocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>)
! FIR:     %[[DECL:.*]] = fir.declare %[[CAST]] uniq_name("_QFs_allocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>
! CHECK:   mif.alloc_coarray %[[DECL]]{{(#0)?}} lcobounds %{{.*}} ucobounds %{{.*}} errmsg %{{.*}} uniq_name("_QFs_allocEc") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
! HLFIR:   hlfir.assign %{{.*}} to %[[DECL]]#0 realloc : !fir.ref<!fir.char<1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>
! HLFIR:   %{{.*}} = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! FIR:     %[[UNCAST:.*]] = fir.volatile_cast %[[DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>>
! FIR:     %[[CONV:.*]] = fir.convert %[[UNCAST]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>>) -> !fir.ref<!fir.box<none>>
! FIR:     fir.call @_FortranAAssign(%[[CONV]], {{.*}})
! CHECK:   mif.dealloc_coarray %[[DECL]]{{(#0)?}} stat %{{.*}} errmsg %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>, volatile, corank:1>, volatile>, !fir.ref<i32>, !fir.box<none>) -> ()

subroutine s_int_alloc()
  integer, volatile, allocatable :: c(:)[:]
  allocate(c(2)[*])
  c = [10, 20]
end subroutine

! CHECK-LABEL: func.func @_QPs_int_alloc()
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>
! CHECK:   %[[CAST:.*]] = fir.volatile_cast %[[ALLOCA]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[CAST]] uniq_name("_QFs_int_allocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>)
! FIR:     %[[DECL:.*]] = fir.declare %[[CAST]] uniq_name("_QFs_int_allocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>
! CHECK:   mif.alloc_coarray %[[DECL]]{{(#0)?}} lcobounds %{{.*}} ucobounds %{{.*}} errmsg %{{.*}} uniq_name("_QFs_int_allocEc") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
! HLFIR:   hlfir.assign %{{.*}} to %[[DECL]]#0 realloc : !fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>
! HLFIR:   %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>
! HLFIR:   %{{.*}} = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:   mif.dealloc_coarray %[[DECL]]#0 stat %{{.*}} errmsg %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>, !fir.ref<i32>, !fir.box<none>) -> ()
! FIR:     %[[EMBOX:.*]] = fir.embox %{{.*}}(%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>
! FIR:     %[[VCAST:.*]] = fir.volatile_cast %[[EMBOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>
! FIR:     fir.store %[[VCAST]] to %[[DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile, corank:1>, volatile>

subroutine s_contiguous()
  character(4), volatile, contiguous, save :: c(2)[*]
  c = "x"
end subroutine

! HLFIR-LABEL: func.func @_QPs_contiguous()
! HLFIR:   %[[VAL_1:.*]] = fir.address_of(@_QFs_contiguousEc) : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>
! HLFIR:   %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %{{.*}} uniq_name("_QFs_contiguousEc") fortran_attrs<contiguous, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>)
! HLFIR:   %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   hlfir.assign %{{.*}} to %[[VAL_6]] : !fir.ref<!fir.char<1>>, !fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>

! FIR-LABEL: func.func @_QPs_contiguous()
! FIR:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<2x!fir.char<1,4>>>
! FIR-NOT:     fir.alloca !fir.box<!fir.heap<
! FIR:         %[[VAL_14:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<2x!fir.char<1,4>>>>) -> !fir.ref<!fir.box<none>>
! FIR:         fir.call @_FortranAAssign(%[[VAL_14]], {{.*}})

subroutine s_alloc_norealloc()
  character(4), volatile, allocatable :: c(:)[:]
  allocate(c(2)[*])
  c = "x"
end subroutine

! CHECK-LABEL: func.func @_QPs_alloc_norealloc()
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>
! CHECK:   %[[CAST:.*]] = fir.volatile_cast %[[ALLOCA]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[CAST]] typeparams %{{.*}} uniq_name("_QFs_alloc_noreallocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>)
! FIR:     %[[DECL:.*]] = fir.declare %[[CAST]] typeparams %{{.*}} uniq_name("_QFs_alloc_noreallocEc") fortran_attrs<allocatable, volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! CHECK:   mif.alloc_coarray %[[DECL]]{{(#0)?}} lcobounds %{{.*}} ucobounds %{{.*}} errmsg %{{.*}} uniq_name("_QFs_alloc_noreallocEc") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
! HLFIR:   hlfir.assign %{{.*}} to %[[DECL]]#0 realloc keep_lhs_len : !fir.ref<!fir.char<1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>
! HLFIR:   %{{.*}} = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>) -> !fir.heap<!fir.array<?x!fir.char<1,4>>>
! HLFIR:   mif.dealloc_coarray %[[DECL]]#0 stat %{{.*}} errmsg %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>, !fir.ref<i32>, !fir.box<none>) -> ()
! FIR:     %[[UNCAST:.*]] = fir.volatile_cast %[[DECL]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>, volatile>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>>
! FIR:     %[[CONV:.*]] = fir.convert %[[UNCAST]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>, volatile, corank:1>>) -> !fir.ref<!fir.box<none>>
! FIR:     fir.call @_FortranAAssignExplicitLengthCharacter(%[[CONV]], {{.*}})

subroutine u()
  integer, volatile, save :: x(2)[2,*]
  print *, ucobound(x)
  print *, ucobound(x, dim=1)
  print *, ucobound(x, dim=2)

end subroutine

! CHECK-LABEL: func.func @_QPu()
! HLFIR:   %[[ADDR:.*]] = fir.address_of(@_QFuEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, corank:2>>
! CHECK:   %[[CAST:.*]] = fir.volatile_cast %{{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, corank:2>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[CAST]] uniq_name("_QFuEx") fortran_attrs<volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>)
! FIR:     %[[DECL:.*]] = fir.declare %[[CAST]] uniq_name("_QFuEx") fortran_attrs<volatile> : (!fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[DECL]]{{(#0)?}} : !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>
! CHECK:   %{{.*}} = mif.ucobound coarray %[[LOAD1]] dim %{{.*}} : (!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, i32) -> i32
! CHECK:   %{{.*}} = mif.ucobound coarray %[[LOAD1]] dim %{{.*}} : (!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, i32) -> i32
! CHECK:   %[[LOAD2:.*]] = fir.load %[[DECL]]{{(#0)?}} : !fir.ref<!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, volatile>
! CHECK:   %{{.*}} = mif.ucobound coarray %[[LOAD2]] dim %{{.*}} : (!fir.box<!fir.heap<!fir.array<2xi32>>, volatile, corank:2>, i32) -> i32
