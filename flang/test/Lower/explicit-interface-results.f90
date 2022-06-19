! RUN: bbc -emit-fir %s -o - | FileCheck %s

module callee
implicit none
contains
! CHECK-LABEL: func @_QMcalleePreturn_cst_array() -> !fir.array<20x30xf32>
function return_cst_array()
  real :: return_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?xf32>
function return_dyn_array(m, n)
  integer :: m, n
  real :: return_dyn_array(m, n)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_cst_array() -> !fir.array<20x30x!fir.char<1,10>>
function return_cst_char_cst_array()
  character(10) :: return_cst_char_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_cst_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<20x30x!fir.char<1,?>>
function return_dyn_char_cst_array(l)
  integer :: l
  character(l) :: return_dyn_char_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?x!fir.char<1,10>>
function return_cst_char_dyn_array(m, n)
  integer :: m, n
  character(10) :: return_cst_char_dyn_array(m, n)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?x!fir.char<1,?>>
function return_dyn_char_dyn_array(l, m, n)
  integer :: l, m, n
  character(l) :: return_dyn_char_dyn_array(m, n)
end function

! CHECK-LABEL: func @_QMcalleePreturn_alloc() -> !fir.box<!fir.heap<!fir.array<?xf32>>>
function return_alloc()
  real, allocatable :: return_alloc(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_alloc() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
function return_cst_char_alloc()
  character(10), allocatable :: return_cst_char_alloc(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_alloc(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
function return_dyn_char_alloc(l)
  integer :: l
  character(l), allocatable :: return_dyn_char_alloc(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_def_char_alloc() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
function return_def_char_alloc()
  character(:), allocatable :: return_def_char_alloc(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_pointer() -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
function return_pointer()
  real, pointer :: return_pointer(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_pointer() -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
function return_cst_char_pointer()
  character(10), pointer :: return_cst_char_pointer(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_pointer(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
function return_dyn_char_pointer(l)
  integer :: l
  character(l), pointer :: return_dyn_char_pointer(:)
end function

! CHECK-LABEL: func @_QMcalleePreturn_def_char_pointer() -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
function return_def_char_pointer()
  character(:), pointer :: return_def_char_pointer(:)
end function
end module

module caller
  use callee
contains

! CHECK-LABEL: func @_QMcallerPcst_array()
subroutine cst_array()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.array<20x30xf32> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, {{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_array() : () -> !fir.array<20x30xf32>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]](%[[shape]]) : !fir.array<20x30xf32>, !fir.ref<!fir.array<20x30xf32>>, !fir.shape<2>
  print *, return_cst_array()
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_cst_array()
subroutine cst_char_cst_array()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.array<20x30x!fir.char<1,10>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, {{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_cst_array() : () -> !fir.array<20x30x!fir.char<1,10>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]](%[[shape]]) typeparams %{{.*}} : !fir.array<20x30x!fir.char<1,10>>, !fir.ref<!fir.array<20x30x!fir.char<1,10>>>, !fir.shape<2>, index
  print *, return_cst_char_cst_array()
end subroutine

! CHECK-LABEL: func @_QMcallerPalloc()
subroutine alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_alloc() : () -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  print *, return_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[cmpi:.*]] = arith.cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]]
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_alloc()
subroutine cst_char_alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_alloc() : () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[cmpi:.*]] = arith.cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]]
end subroutine

! CHECK-LABEL: func @_QMcallerPdef_char_alloc()
subroutine def_char_alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_def_char_alloc() : () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[cmpi:.*]] = arith.cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]]
end subroutine

! CHECK-LABEL: func @_QMcallerPpointer_test()
subroutine pointer_test()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  print *, return_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_pointer()
subroutine cst_char_pointer()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func @_QMcallerPdef_char_pointer()
subroutine def_char_pointer()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_def_char_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func @_QMcallerPdyn_array(
! CHECK-SAME: %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_array(m, n)
  integer :: m, n
  ! CHECK-DAG: %[[mload:.*]] = fir.load %[[m]] : !fir.ref<i32>
  ! CHECK-DAG: %[[mcast:.*]] = fir.convert %[[mload]] : (i32) -> i64
  ! CHECK-DAG: %[[msub:.*]] = arith.subi %[[mcast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[madd:.*]] = arith.addi %[[msub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[mcast2:.*]] = fir.convert %[[madd]] : (i64) -> index
  ! CHECK-DAG: %[[nload:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast:.*]] = fir.convert %[[nload]] : (i32) -> i64
  ! CHECK-DAG: %[[nsub:.*]] = arith.subi %[[ncast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[nadd:.*]] = arith.addi %[[nsub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[ncast2:.*]] = fir.convert %[[nadd]] : (i64) -> index
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.array<?x?xf32>, %[[mcast2]], %[[ncast2]]
  ! CHECK: %[[shape:.*]] = fir.shape %[[mcast2]], %[[ncast2]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_dyn_array(%[[m]], %[[n]]) : (!fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?xf32>
  ! CHECK: fir.save_result %[[res]] to %[[tmp]](%[[shape]]) : !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>
  print *, return_dyn_array(m, n)
end subroutine

! CHECK-LABEL: func @_QMcallerPdyn_char_cst_array(
! CHECK-SAME: %[[l:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_char_cst_array(l)
  integer :: l
  ! CHECK: %[[lload:.*]] = fir.load %[[l]] : !fir.ref<i32>
  ! CHECK: %[[lcast:.*]] = fir.convert %[[lload]] : (i32) -> i64
  ! CHECK: %[[lcast2:.*]] = fir.convert %[[lcast]] : (i64) -> index
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.array<20x30x!fir.char<1,?>>(%[[lcast2]] : index)
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_dyn_char_cst_array(%[[l]]) : (!fir.ref<i32>) -> !fir.array<20x30x!fir.char<1,?>>
  ! CHECK: fir.save_result %[[res]] to %[[tmp]](%[[shape]]) typeparams %[[lcast2]] : !fir.array<20x30x!fir.char<1,?>>, !fir.ref<!fir.array<20x30x!fir.char<1,?>>>, !fir.shape<2>, index
  print *, return_dyn_char_cst_array(l)
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_dyn_array(
! CHECK-SAME: %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine cst_char_dyn_array(m, n)
  integer :: m, n
  ! CHECK-DAG: %[[mload:.*]] = fir.load %[[m]] : !fir.ref<i32>
  ! CHECK-DAG: %[[mcast:.*]] = fir.convert %[[mload]] : (i32) -> i64
  ! CHECK-DAG: %[[msub:.*]] = arith.subi %[[mcast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[madd:.*]] = arith.addi %[[msub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[mcast2:.*]] = fir.convert %[[madd]] : (i64) -> index
  ! CHECK-DAG: %[[nload:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast:.*]] = fir.convert %[[nload]] : (i32) -> i64
  ! CHECK-DAG: %[[nsub:.*]] = arith.subi %[[ncast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[nadd:.*]] = arith.addi %[[nsub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[ncast2:.*]] = fir.convert %[[nadd]] : (i64) -> index
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,10>>, %[[mcast2]], %[[ncast2]]
  ! CHECK: %[[shape:.*]] = fir.shape %[[mcast2]], %[[ncast2]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_dyn_array(%[[m]], %[[n]]) : (!fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,10>>
  ! CHECK: fir.save_result %[[res]] to %[[tmp]](%[[shape]]) typeparams {{.*}} : !fir.array<?x?x!fir.char<1,10>>, !fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index
  print *, return_cst_char_dyn_array(m, n)
end subroutine

! CHECK-LABEL: func @_QMcallerPdyn_char_dyn_array(
! CHECK-SAME: %[[l:.*]]: !fir.ref<i32>{{.*}}, %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_char_dyn_array(l, m, n)
  ! CHECK-DAG: %[[mload:.*]] = fir.load %[[m]] : !fir.ref<i32>
  ! CHECK-DAG: %[[mcast:.*]] = fir.convert %[[mload]] : (i32) -> i64
  ! CHECK-DAG: %[[msub:.*]] = arith.subi %[[mcast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[madd:.*]] = arith.addi %[[msub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[mcast2:.*]] = fir.convert %[[madd]] : (i64) -> index

  ! CHECK-DAG: %[[nload:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK-DAG: %[[ncast:.*]] = fir.convert %[[nload]] : (i32) -> i64
  ! CHECK-DAG: %[[nsub:.*]] = arith.subi %[[ncast]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[nadd:.*]] = arith.addi %[[nsub]], %c1{{.*}} : i64
  ! CHECK-DAG: %[[ncast2:.*]] = fir.convert %[[nadd]] : (i64) -> index

  ! CHECK-DAG: %[[lload:.*]] = fir.load %[[l]] : !fir.ref<i32>
  ! CHECK-DAG: %[[lcast:.*]] = fir.convert %[[lload]] : (i32) -> i64
  ! CHECK-DAG: %[[lcast2:.*]] = fir.convert %[[lcast]] : (i64) -> index
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%[[lcast2]] : index), %[[mcast2]], %[[ncast2]]
  ! CHECK: %[[shape:.*]] = fir.shape %[[mcast2]], %[[ncast2]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_dyn_char_dyn_array(%[[l]], %[[m]], %[[n]]) : (!fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,?>>
  ! CHECK: fir.save_result %[[res]] to %[[tmp]](%[[shape]]) typeparams {{.*}} : !fir.array<?x?x!fir.char<1,?>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index
  integer :: l, m, n
  print *, return_dyn_char_dyn_array(l, m, n)
end subroutine

! CHECK-LABEL: @_QMcallerPdyn_char_alloc
subroutine dyn_char_alloc(l)
  integer :: l
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_dyn_char_alloc({{.*}}) : (!fir.ref<i32>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_dyn_char_alloc(l)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[cmpi:.*]] = arith.cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]]
end subroutine

! CHECK-LABEL: @_QMcallerPdyn_char_pointer
subroutine dyn_char_pointer(l)
  integer :: l
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {{{.*}}bindc_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_dyn_char_pointer({{.*}}) : (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_dyn_char_pointer(l)
  ! CHECK-NOT: fir.freemem
end subroutine

end module


! Test more complex symbol dependencies in the result specification expression

module m_with_equiv
  integer(8) :: l
  integer(8) :: array(3)
  equivalence (array(2), l)
contains
  function result_depends_on_equiv_sym()
    character(l) :: result_depends_on_equiv_sym
    call set_result_with_some_value(result_depends_on_equiv_sym)
  end function  
end module

! CHECK-LABEL: func @_QPtest_result_depends_on_equiv_sym
subroutine test_result_depends_on_equiv_sym()
  use m_with_equiv, only : result_depends_on_equiv_sym
  ! CHECK: %[[equiv:.*]] = fir.address_of(@_QMm_with_equivEarray) : !fir.ref<!fir.array<24xi8>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[equiv]], %c{{.*}} : (!fir.ref<!fir.array<24xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[l:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<i64>
  ! CHECK: %[[load:.*]] = fir.load %[[l]] : !fir.ptr<i64>
  ! CHECK: %[[lcast:.*]] = fir.convert %[[load]] : (i64) -> index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[lcast]] : index)
  print *, result_depends_on_equiv_sym()
end subroutine

! CHECK-LABEL: func @_QPtest_depends_on_descriptor(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_depends_on_descriptor(x)
  interface
    function depends_on_descriptor(x)
      real :: x(:)
      character(size(x,1, KIND=8)) :: depends_on_descriptor
    end function
  end interface
  real :: x(:)
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK: %[[extentCast:.*]] = fir.convert %[[dims]]#1 : (index) -> i64
  ! CHECK: %[[extent:.*]] = fir.convert %[[extentCast]] : (i64) -> index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[extent]] : index)
  print *, depends_on_descriptor(x)
end subroutine

! CHECK-LABEL: func @_QPtest_symbol_indirection(
! CHECK-SAME: %[[n:.*]]: !fir.ref<i64>{{.*}}) {
subroutine test_symbol_indirection(n)
  interface
    function symbol_indirection(c, n)
      integer(8) :: n
      character(n) :: c
      character(len(c, KIND=8)) :: symbol_indirection
    end function
  end interface
  integer(8) :: n
  character(n) :: c
  ! CHECK: BeginExternalListOutput
  ! CHECK: %[[nload:.*]] = fir.load %[[n]] : !fir.ref<i64>
  ! CHECK: %[[n_is_positive:.*]] = arith.cmpi sgt, %[[nload]], %c0{{.*}} : i64
  ! CHECK: %[[len:.*]] = arith.select %[[n_is_positive]], %[[nload]], %c0{{.*}} : i64
  ! CHECK: %[[len_cast:.*]] = fir.convert %[[len]] : (i64) -> index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[len_cast]] : index)
  print *, symbol_indirection(c, n)
end subroutine

! CHECK-LABEL: func @_QPtest_recursion(
! CHECK-SAME: %[[res:.*]]: !fir.ref<!fir.char<1,?>>{{.*}}, %[[resLen:.*]]: index{{.*}}, %[[n:.*]]: !fir.ref<i64>{{.*}}) -> !fir.boxchar<1> {
function test_recursion(n) result(res)
  integer(8) :: n
  character(n) :: res
  ! some_local is here to verify that local symbols that are visible in the
  ! function interface are not instantiated by accident (that only the
  ! symbols needed for the result are instantiated before the call).
  ! CHECK: fir.alloca !fir.array<?xi32>, {{.*}}some_local
  ! CHECK-NOT: fir.alloca !fir.array<?xi32>
  integer :: some_local(n)
  some_local(0) = n + 64
  if (n.eq.1) then
    res = char(some_local(0))
  ! CHECK: else
  else 
    ! CHECK-NOT: fir.alloca !fir.array<?xi32>

    ! verify that the actual argument for symbol n ("n-1") is used to allocate
    ! the result, and not the local value of symbol n.

    ! CHECK: %[[nLoad:.*]] = fir.load %[[n]] : !fir.ref<i64>
    ! CHECK: %[[sub:.*]] = arith.subi %[[nLoad]], %c1{{.*}} : i64
    ! CHECK: fir.store %[[sub]] to %[[nInCall:.*]] : !fir.ref<i64>

    ! CHECK-NOT: fir.alloca !fir.array<?xi32>

    ! CHECK: %[[nInCallLoad:.*]] = fir.load %[[nInCall]] : !fir.ref<i64>
    ! CHECK: %[[nInCallCast:.*]] = fir.convert %[[nInCallLoad]] : (i64) -> index
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.char<1,?>(%[[nInCallCast]] : index)

    ! CHECK-NOT: fir.alloca !fir.array<?xi32>
    ! CHECK: fir.call @_QPtest_recursion(%[[tmp]], {{.*}}
    res = char(some_local(0)) // test_recursion(n-1)

    ! Verify that symbol n was not remapped to the actual argument passed
    ! to n in the call (that the temporary mapping was cleaned-up).

    ! CHECK: %[[nLoad2:.*]] = fir.load %[[n]] : !fir.ref<i64>
    ! CHECK: OutputInteger64(%{{.*}}, %[[nLoad2]])
    print *, n
  end if
end function

! Test call to character function for which only the result type is explicit
! CHECK-LABEL:func @_QPtest_not_entirely_explicit_interface(
! CHECK-SAME: %[[n_arg:.*]]: !fir.ref<i64>{{.*}}) {
subroutine test_not_entirely_explicit_interface(n)
  integer(8) :: n
  character(n) :: return_dyn_char_2
  print *, return_dyn_char_2(10)
  ! CHECK: %[[n:.*]] = fir.load %[[n_arg]] : !fir.ref<i64>
  ! CHECK: %[[len:.*]] = fir.convert %[[n]] : (i64) -> index
  ! CHECK: %[[result:.*]] = fir.alloca !fir.char<1,?>(%[[len]] : index) {bindc_name = ".result"}
  ! CHECK: fir.call @_QPreturn_dyn_char_2(%[[result]], %[[len]], %{{.*}}) : (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
end subroutine
