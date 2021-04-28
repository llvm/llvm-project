! RUN: bbc -emit-fir %s -o - | FileCheck %s

module callee
implicit none
contains
! CHECK-LABEL: func @_QMcalleePreturn_cst_array() -> !fir.array<20x30xf32>
function return_cst_array()
  real :: return_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_array(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>) -> !fir.array<?x?xf32>
function return_dyn_array(m, n)
  integer :: m, n
  real :: return_dyn_array(m, n)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_cst_array() -> !fir.array<20x30x!fir.char<1,10>>
function return_cst_char_cst_array()
  character(10) :: return_cst_char_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_cst_array(%{{.*}}: !fir.ref<i32>) -> !fir.array<20x30x!fir.char<1,?>>
function return_dyn_char_cst_array(l)
  integer :: l
  character(l) :: return_dyn_char_cst_array(20, 30)
end function

! CHECK-LABEL: func @_QMcalleePreturn_cst_char_dyn_array(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,10>>
function return_cst_char_dyn_array(m, n)
  integer :: m, n
  character(10) :: return_cst_char_dyn_array(m, n)
end function

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_dyn_array(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>) -> !fir.array<?x?x!fir.char<1,?>>
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

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_alloc(%{{.*}}: !fir.ref<i32>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
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

! CHECK-LABEL: func @_QMcalleePreturn_dyn_char_pointer(%{{.*}}: !fir.ref<i32>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
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
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.array<20x30xf32> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, {{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_array() : () -> !fir.array<20x30xf32>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]](%[[shape]]) : !fir.array<20x30xf32>, !fir.ref<!fir.array<20x30xf32>>, !fir.shape<2>
  print *, return_cst_array()
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_cst_array()
subroutine cst_char_cst_array()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.array<20x30x!fir.char<1,10>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}}, {{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_cst_array() : () -> !fir.array<20x30x!fir.char<1,10>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]](%[[shape]]) typeparams %{{.*}} : !fir.array<20x30x!fir.char<1,10>>, !fir.ref<!fir.array<20x30x!fir.char<1,10>>>, !fir.shape<2>, index
  print *, return_cst_char_cst_array()
end subroutine

! CHECK-LABEL: func @_QMcallerPalloc()
subroutine alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_alloc() : () -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  print *, return_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[cmpi:.*]] = cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_alloc()
subroutine cst_char_alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_alloc() : () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[cmpi:.*]] = cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
end subroutine

! CHECK-LABEL: func @_QMcallerPdef_char_alloc()
subroutine def_char_alloc()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_def_char_alloc() : () -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_alloc()
  ! CHECK: _FortranAioOutputDescriptor
  ! CHECK: %[[load:.*]] = fir.load %[[alloc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[cmpi:.*]] = cmpi
  ! CHECK: fir.if %[[cmpi]]
  ! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
end subroutine

! CHECK-LABEL: func @_QMcallerPpointer_test()
subroutine pointer_test()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  print *, return_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func @_QMcallerPcst_char_pointer()
subroutine cst_char_pointer()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_cst_char_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
  print *, return_cst_char_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! CHECK-LABEL: func @_QMcallerPdef_char_pointer()
subroutine def_char_pointer()
  ! CHECK: %[[alloc:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {{{.*}}uniq_name = ".result"}
  ! CHECK: %[[res:.*]] = fir.call @_QMcalleePreturn_def_char_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.save_result %[[res]] to %[[alloc]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  print *, return_def_char_pointer()
  ! CHECK-NOT: fir.freemem
end subroutine

! TODO: dynamic specification expression in results
!subroutine dyn_array(m, n)
!  integer :: m, n
!  print *, return_dyn_array(m, n)
!end subroutine
!subroutine dyn_char_cst_array(l)
!  integer :: l
!  print *, return_dyn_char_cst_array(l)
!end subroutine
!subroutine cst_char_dyn_array(m, n)
!  integer :: m, n
!  print *, return_cst_char_dyn_array(m, n)
!end subroutine
!subroutine dyn_char_dyn_array(l, m, n)
!  integer :: l, m, n
!  print *, return_dyn_char_dyn_array(l, m, n)
!end subroutine
!subroutine dyn_char_alloc(l)
!  integer :: l
!  print *, return_dyn_char_alloc(l)
!end subroutine
!subroutine dyn_char_pointer(l)
!  integer :: l
!  print *, return_dyn_char_pointer(l)
!end subroutine

end module
