! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module callee
implicit none
contains
! CHECK-LABEL: func.func @_QMcalleePreturn_cst_array() -> !fir.array<20x30xf32>
function return_cst_array()
  real :: return_cst_array(20, 30)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?xf32>
function return_dyn_array(m, n)
  integer :: m, n
  real :: return_dyn_array(m, n)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_cst_char_cst_array() -> !fir.array<20x30x!fir.char<1,10>>
function return_cst_char_cst_array()
  character(10) :: return_cst_char_cst_array(20, 30)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_dyn_char_cst_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<20x30x!fir.char<1,?>>
function return_dyn_char_cst_array(l)
  integer :: l
  character(l) :: return_dyn_char_cst_array(20, 30)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_cst_char_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?x!fir.char<1,10>>
function return_cst_char_dyn_array(m, n)
  integer :: m, n
  character(10) :: return_cst_char_dyn_array(m, n)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_dyn_char_dyn_array(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.array<?x?x!fir.char<1,?>>
function return_dyn_char_dyn_array(l, m, n)
  integer :: l, m, n
  character(l) :: return_dyn_char_dyn_array(m, n)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_alloc() -> !fir.box<!fir.heap<!fir.array<?xf32>>>
function return_alloc()
  real, allocatable :: return_alloc(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_cst_char_alloc() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
function return_cst_char_alloc()
  character(10), allocatable :: return_cst_char_alloc(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_dyn_char_alloc(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
function return_dyn_char_alloc(l)
  integer :: l
  character(l), allocatable :: return_dyn_char_alloc(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_def_char_alloc() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
function return_def_char_alloc()
  character(:), allocatable :: return_def_char_alloc(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_pointer() -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
function return_pointer()
  real, pointer :: return_pointer(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_cst_char_pointer() -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
function return_cst_char_pointer()
  character(10), pointer :: return_cst_char_pointer(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_dyn_char_pointer(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
function return_dyn_char_pointer(l)
  integer :: l
  character(l), pointer :: return_dyn_char_pointer(:)
end function

! CHECK-LABEL: func.func @_QMcalleePreturn_def_char_pointer() -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
function return_def_char_pointer()
  character(:), pointer :: return_def_char_pointer(:)
end function
end module

module caller
  use callee
contains

! CHECK-LABEL: func.func @_QMcallerPcst_array()
subroutine cst_array()
  ! CHECK: %[[VAL_14:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_15:.*]] = hlfir.eval_in_mem shape %[[VAL_14]] : (!fir.shape<2>) -> !hlfir.expr<20x30xf32> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<20x30xf32>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_cst_array() {{.*}}: () -> !fir.array<20x30xf32>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_14]]) : !fir.array<20x30xf32>, !fir.ref<!fir.array<20x30xf32>>, !fir.shape<2>
  ! CHECK: }
  ! CHECK: %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_15]](%[[VAL_14]])
  print *, return_cst_array()
end subroutine

! CHECK-LABEL: func.func @_QMcallerPcst_char_cst_array()
subroutine cst_char_cst_array()
  ! CHECK: %[[VAL_17:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_18:.*]] = hlfir.eval_in_mem shape %[[VAL_17]] typeparams %{{.*}} : (!fir.shape<2>, index) -> !hlfir.expr<20x30x!fir.char<1,10>> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<20x30x!fir.char<1,10>>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_cst_char_cst_array() {{.*}}: () -> !fir.array<20x30x!fir.char<1,10>>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_17]]) typeparams %{{.*}} : !fir.array<20x30x!fir.char<1,10>>, !fir.ref<!fir.array<20x30x!fir.char<1,10>>>, !fir.shape<2>, index
  ! CHECK: }
  ! CHECK: %[[VAL_19:.*]]:3 = hlfir.associate %[[VAL_18]](%[[VAL_17]]) typeparams %{{.*}}
  print *, return_cst_char_cst_array()
end subroutine

! CHECK-LABEL: func.func @_QMcallerPalloc()
subroutine alloc()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"}
  ! CHECK: %[[VAL_6:.*]] = fir.call @_QMcalleePreturn_alloc() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[VAL_6]] to %[[VAL_5]]#0 : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  print *, return_alloc()
  ! CHECK: %[[load:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[as_expr:.*]] = hlfir.as_expr %[[load]] move %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, i1) -> !hlfir.expr<?xf32>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[as_expr]]({{.*}}) {adapt.valuebyref} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?xf32>>, i1
  ! CHECK: hlfir.destroy %[[as_expr]] : !hlfir.expr<?xf32>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPcst_char_alloc()
subroutine cst_char_alloc()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %{{.*}} {uniq_name = ".tmp.func_result"}
  ! CHECK: %[[VAL_10:.*]] = fir.call @_QMcalleePreturn_cst_char_alloc() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[VAL_10]] to %[[VAL_9]]#0 : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_alloc()
  ! CHECK: %[[load:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: %[[as_expr:.*]] = hlfir.as_expr %[[load]] move %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, i1) -> !hlfir.expr<?x!fir.char<1,10>>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[as_expr]]({{.*}}) typeparams %{{.*}} {adapt.valuebyref} : (!hlfir.expr<?x!fir.char<1,10>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>, i1)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?x!fir.char<1,10>>>, i1
  ! CHECK: hlfir.destroy %[[as_expr]] : !hlfir.expr<?x!fir.char<1,10>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdef_char_alloc()
subroutine def_char_alloc()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = ".tmp.func_result"}
  ! CHECK: %[[VAL_6:.*]] = fir.call @_QMcalleePreturn_def_char_alloc() {{.*}}: () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[VAL_6]] to %[[VAL_5]]#0 : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_alloc()
  ! CHECK: %[[load:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[as_expr:.*]] = hlfir.as_expr %[[load]] move %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, i1) -> !hlfir.expr<?x!fir.char<1,?>>
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[as_expr]]({{.*}}) typeparams %{{.*}} {adapt.valuebyref} : (!hlfir.expr<?x!fir.char<1,?>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<!fir.array<?x!fir.char<1,?>>>, i1
  ! CHECK: hlfir.destroy %[[as_expr]] : !hlfir.expr<?x!fir.char<1,?>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPpointer_test()
subroutine pointer_test()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_5:.*]] = fir.call @_QMcalleePreturn_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[VAL_5]] to %[[VAL_0]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  print *, return_pointer()
  ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func.func @_QMcallerPcst_char_pointer()
subroutine cst_char_pointer()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_9:.*]] = fir.call @_QMcalleePreturn_cst_char_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[VAL_9]] to %[[VAL_0]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_pointer()
  ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdef_char_pointer()
subroutine def_char_pointer()
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_5:.*]] = fir.call @_QMcalleePreturn_def_char_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[VAL_5]] to %[[VAL_0]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_pointer()
  ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdyn_array(
! CHECK-SAME: %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_array(m, n)
  integer :: m, n
  ! CHECK: %[[VAL_22:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_23:.*]] = hlfir.eval_in_mem shape %[[VAL_22]] : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?x?xf32>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_dyn_array(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?xf32>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_22]]) : !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>
  ! CHECK: }
  ! CHECK: %[[VAL_24:.*]]:3 = hlfir.associate %[[VAL_23]](%[[VAL_22]]) {adapt.valuebyref}
  print *, return_dyn_array(m, n)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[VAL_24]]#1, %[[VAL_24]]#2 : !fir.ref<!fir.array<?x?xf32>>, i1
  ! CHECK: hlfir.destroy %[[VAL_23]] : !hlfir.expr<?x?xf32>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdyn_char_cst_array(
! CHECK-SAME: %[[l:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_char_cst_array(l)
  integer :: l
  ! CHECK: %[[VAL_21:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_22:.*]] = hlfir.eval_in_mem shape %[[VAL_21]] typeparams %[[VAL_20:.*]] : (!fir.shape<2>, index) -> !hlfir.expr<20x30x!fir.char<1,?>> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<20x30x!fir.char<1,?>>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_dyn_char_cst_array(%{{.*}}) {{.*}}: (!fir.ref<i32>) -> !fir.array<20x30x!fir.char<1,?>>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_21]]) typeparams %[[VAL_20]] : !fir.array<20x30x!fir.char<1,?>>, !fir.ref<!fir.array<20x30x!fir.char<1,?>>>, !fir.shape<2>, index
  ! CHECK: }
  ! CHECK: %[[VAL_23:.*]]:3 = hlfir.associate %[[VAL_22]](%[[VAL_21]]) typeparams %[[VAL_20]] {adapt.valuebyref}
  print *, return_dyn_char_cst_array(l)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[VAL_23]]#1, %[[VAL_23]]#2 : !fir.ref<!fir.array<20x30x!fir.char<1,?>>>, i1
  ! CHECK: hlfir.destroy %[[VAL_22]] : !hlfir.expr<20x30x!fir.char<1,?>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPcst_char_dyn_array(
! CHECK-SAME: %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine cst_char_dyn_array(m, n)
  integer :: m, n
  ! CHECK: %[[VAL_25:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_26:.*]] = hlfir.eval_in_mem shape %[[VAL_25]] typeparams %[[VAL_24:.*]] : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<1,10>> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?x?x!fir.char<1,10>>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_cst_char_dyn_array(%{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,10>>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_25]]) typeparams %[[VAL_24]] : !fir.array<?x?x!fir.char<1,10>>, !fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index
  ! CHECK: }
  ! CHECK: %[[VAL_27:.*]]:3 = hlfir.associate %[[VAL_26]](%[[VAL_25]]) typeparams %[[VAL_24]] {adapt.valuebyref}
  print *, return_cst_char_dyn_array(m, n)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[VAL_27]]#1, %[[VAL_27]]#2 : !fir.ref<!fir.array<?x?x!fir.char<1,10>>>, i1
  ! CHECK: hlfir.destroy %[[VAL_26]] : !hlfir.expr<?x?x!fir.char<1,10>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdyn_char_dyn_array(
! CHECK-SAME: %[[l:.*]]: !fir.ref<i32>{{.*}}, %[[m:.*]]: !fir.ref<i32>{{.*}}, %[[n:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dyn_char_dyn_array(l, m, n)
  ! CHECK: %[[VAL_29:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_30:.*]] = hlfir.eval_in_mem shape %[[VAL_29]] typeparams %[[VAL_28:.*]] : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<1,?>> {
  ! CHECK: ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?x?x!fir.char<1,?>>>):
  ! CHECK:   %[[RES:.*]] = fir.call @_QMcalleePreturn_dyn_char_dyn_array(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,?>>
  ! CHECK:   fir.save_result %[[RES]] to %[[ARG]](%[[VAL_29]]) typeparams %[[VAL_28]] : !fir.array<?x?x!fir.char<1,?>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index
  ! CHECK: }
  ! CHECK: %[[VAL_31:.*]]:3 = hlfir.associate %[[VAL_30]](%[[VAL_29]]) typeparams %[[VAL_28]] {adapt.valuebyref}
  integer :: l, m, n
  print *, return_dyn_char_dyn_array(l, m, n)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.end_associate %[[VAL_31]]#1, %[[VAL_31]]#2 : !fir.ref<!fir.array<?x?x!fir.char<1,?>>>, i1
  ! CHECK: hlfir.destroy %[[VAL_30]] : !hlfir.expr<?x?x!fir.char<1,?>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdyn_char_alloc
subroutine dyn_char_alloc(l)
  integer :: l
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_11:.*]] {uniq_name = ".tmp.func_result"}
  ! CHECK: %[[VAL_14:.*]] = fir.call @_QMcalleePreturn_dyn_char_alloc({{.*}}) {{.*}}: (!fir.ref<i32>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[VAL_14]] to %[[VAL_13]]#0 : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_dyn_char_alloc(l)
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: hlfir.destroy %{{.*}} : !hlfir.expr<?x!fir.char<1,?>>
end subroutine

! CHECK-LABEL: func.func @_QMcallerPdyn_char_pointer
subroutine dyn_char_pointer(l)
  integer :: l
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {bindc_name = ".result"}
  ! CHECK: %[[VAL_13:.*]] = fir.call @_QMcalleePreturn_dyn_char_pointer({{.*}}) {{.*}}: (!fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[VAL_13]] to %[[VAL_0]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_11:.*]] {uniq_name = ".tmp.func_result"}
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

! CHECK-LABEL: func.func @_QPtest_result_depends_on_equiv_sym
subroutine test_result_depends_on_equiv_sym()
  use m_with_equiv, only : result_depends_on_equiv_sym
  ! CHECK: %[[equiv:.*]] = fir.address_of(@_QMm_with_equivEarray) : !fir.ref<!fir.array<24xi8>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[equiv]], %c{{.*}} : (!fir.ref<!fir.array<24xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[l:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<i64>
  ! CHECK: %[[l_decl:.*]]:2 = hlfir.declare %[[l]] storage(%[[equiv]][8]) {uniq_name = "_QMm_with_equivEl"}
  ! CHECK: %[[load:.*]] = fir.load %[[l_decl]]#0 : !fir.ptr<i64>
  ! CHECK: %[[lcast:.*]] = fir.convert %[[load]] : (i64) -> index
  ! CHECK: %[[cmpi:.*]] = arith.cmpi sgt, %[[lcast]], %{{.*}} : index
  ! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[lcast]], %{{.*}} : index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[select]] : index)
  print *, result_depends_on_equiv_sym()
end subroutine

! CHECK-LABEL: func.func @_QPtest_depends_on_descriptor(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_depends_on_descriptor(x)
  interface
    function depends_on_descriptor(x)
      real :: x(:)
      character(size(x,1, KIND=8)) :: depends_on_descriptor
    end function
  end interface
  real :: x(:)
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %{{.*}}, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK: %[[extentCast:.*]] = fir.convert %[[dims]]#1 : (index) -> i64
  ! CHECK: %[[extent:.*]] = fir.convert %[[extentCast]] : (i64) -> index
  ! CHECK: %[[cmpi:.*]] = arith.cmpi sgt, %[[extent]], %{{.*}} : index
  ! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[extent]], %{{.*}} : index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[select]] : index)
  print *, depends_on_descriptor(x)
end subroutine

! CHECK-LABEL: func.func @_QPtest_symbol_indirection(
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
  ! CHECK: %[[nload:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK: %[[n_is_positive:.*]] = arith.cmpi sgt, %[[nload]], %c0{{.*}} : i64
  ! CHECK: %[[len:.*]] = arith.select %[[n_is_positive]], %[[nload]], %c0{{.*}} : i64
  ! CHECK: %[[len_cast:.*]] = fir.convert %[[len]] : (i64) -> index
  ! CHECK: %[[cmpi:.*]] = arith.cmpi sgt, %[[len_cast]], %{{.*}} : index
  ! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[len_cast]], %{{.*}} : index
  ! CHECK: fir.alloca !fir.char<1,?>(%[[select]] : index)
  print *, symbol_indirection(c, n)
end subroutine

! CHECK-LABEL: func.func @_QPtest_recursion(
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
  some_local(1) = n + 64
  if (n.eq.1) then
    res = char(some_local(1))
  ! CHECK: else
  else
    ! CHECK-NOT: fir.alloca !fir.array<?xi32>

    ! verify that the actual argument for symbol n ("n-1") is used to allocate
    ! the result, and not the local value of symbol n.

    ! CHECK: %[[nLoad:.*]] = fir.load %[[n_decl:.*]]#0 : !fir.ref<i64>
    ! CHECK: %[[sub:.*]] = arith.subi %[[nLoad]], %c1{{.*}} : i64
    ! CHECK: %[[nInCall_assoc:.*]]:3 = hlfir.associate %[[sub]] {adapt.valuebyref} : (i64) -> (!fir.ref<i64>, !fir.ref<i64>, i1)
    ! CHECK: %[[nInCall_decl:.*]]:2 = hlfir.declare %[[nInCall_assoc]]#0 {uniq_name = "_QFtest_recursionEn"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)

    ! CHECK-NOT: fir.alloca !fir.array<?xi32>

    ! CHECK: %[[nInCallLoad:.*]] = fir.load %[[nInCall_decl]]#0 : !fir.ref<i64>
    ! CHECK: %[[nInCallCast:.*]] = fir.convert %[[nInCallLoad]] : (i64) -> index
    ! CHECK: %[[cmpi:.*]] = arith.cmpi sgt, %[[nInCallCast]], %{{.*}} : index
    ! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[nInCallCast]], %{{.*}} : index
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.char<1,?>(%[[select]] : index)

    ! CHECK-NOT: fir.alloca !fir.array<?xi32>
    ! CHECK: fir.call @_QPtest_recursion(%[[tmp]], %[[select]], %[[nInCall_assoc]]#0) {{.*}}
    res = char(some_local(1)) // test_recursion(n-1)
    ! CHECK: hlfir.end_associate %[[nInCall_assoc]]#1, %[[nInCall_assoc]]#2 : !fir.ref<i64>, i1

    ! Verify that symbol n was not remapped to the actual argument passed
    ! to n in the call (that the temporary mapping was cleaned-up).

    ! CHECK: %[[nLoad2:.*]] = fir.load %[[n_decl]]#0 : !fir.ref<i64>
    ! CHECK: OutputInteger64(%{{.*}}, %[[nLoad2]])
    print *, n
  end if
end function

! Test call to character function for which only the result type is explicit
! CHECK-LABEL:func.func @_QPtest_not_entirely_explicit_interface(
! CHECK-SAME: %[[n_arg:.*]]: !fir.ref<i64>{{.*}}) {
subroutine test_not_entirely_explicit_interface(n)
  integer(8) :: n
  character(n) :: return_dyn_char_2
  print *, return_dyn_char_2(10)
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %c10_i32 {adapt.valuebyref}
  ! CHECK: %[[n:.*]] = fir.load %[[n_decl:.*]]#0 : !fir.ref<i64>
  ! CHECK: %[[len:.*]] = fir.convert %[[n]] : (i64) -> index
  ! CHECK: %[[cmpi:.*]] = arith.cmpi sgt, %[[len]], %{{.*}} : index
  ! CHECK: %[[select:.*]] = arith.select %[[cmpi]], %[[len]], %{{.*}} : index
  ! CHECK: %[[result:.*]] = fir.alloca !fir.char<1,?>(%[[select]] : index) {bindc_name = ".result"}
  ! CHECK: fir.call @_QPreturn_dyn_char_2(%[[result]], %[[select]], %[[assoc]]#0) {{.*}}: (!fir.ref<!fir.char<1,?>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2 : !fir.ref<i32>, i1
end subroutine
