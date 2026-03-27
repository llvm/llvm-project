! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module some_module
 integer :: n_module
end module

! Test host calling array internal procedure.
! Result depends on host variable.
! CHECK-LABEL: func.func @_QPhost1
subroutine host1()
  implicit none
  integer :: n
  call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]] = hlfir.eval_in_mem shape %[[VAL_7:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost1Preturn_array(%{{.*}}) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_7]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]](%[[VAL_7]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_9]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
contains
  function return_array()
    real :: return_array(n)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on module variable with the use statement inside the host.
! CHECK-LABEL: func.func @_QPhost2
subroutine host2()
  use :: some_module
  call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.eval_in_mem shape %[[VAL_4:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost2Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_4]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_4]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
contains
  function return_array()
    real :: return_array(n_module)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on module variable with the use statement inside the internal procedure.
! CHECK-LABEL: func.func @_QPhost3
subroutine host3()
  call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.eval_in_mem shape %[[VAL_4:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost3Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_4]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_4]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
contains
  function return_array()
    use :: some_module
    real :: return_array(n_module)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on host variable not directly used in A.
subroutine host4()
  implicit none
  integer :: n
  call internal_proc_a()
contains
! CHECK-LABEL: func.func private @_QFhost4Pinternal_proc_a
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc})
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]] = hlfir.eval_in_mem shape %[[VAL_7:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost4Preturn_array(%[[VAL_0]]) {{.*}}: (!fir.ref<tuple<!fir.ref<i32>>>) -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_7]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_8]](%[[VAL_7]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_9]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  end subroutine
  function return_array()
    real :: return_array(n)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on module variable with use statement in the host.
subroutine host5()
  use :: some_module
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func.func private @_QFhost5Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.eval_in_mem shape %[[VAL_4:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost5Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_4]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_4]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  end subroutine
  function return_array()
    real :: return_array(n_module)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on module variable with use statement in B.
subroutine host6()
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func.func private @_QFhost6Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = hlfir.eval_in_mem shape %[[VAL_4:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost6Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_4]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_5]](%[[VAL_4]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_6]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  end subroutine
  function return_array()
    use :: some_module
    real :: return_array(n_module)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on a common block variable declared in the host.
! CHECK-LABEL: func.func @_QPhost7
subroutine host7()
  implicit none
  integer :: n_common
  common /mycom/ n_common
  call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_12:.*]] = hlfir.eval_in_mem shape %[[VAL_11:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost7Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_11]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_13:.*]]:3 = hlfir.associate %[[VAL_12]](%[[VAL_11]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_13]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
contains
  function return_array()
    real :: return_array(n_common)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on a common block variable declared in the internal procedure.
! CHECK-LABEL: func.func @_QPhost8
subroutine host8()
  implicit none
  call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]] = hlfir.eval_in_mem shape %[[VAL_8:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost8Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_8]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_9]](%[[VAL_8]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_10]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
contains
  function return_array()
    integer :: n_common
    common /mycom/ n_common
    real :: return_array(n_common)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on a common block variable declared in the host.
subroutine host9()
  implicit none
  integer :: n_common
  common /mycom/ n_common
  call internal_proc_a()
contains
! CHECK-LABEL: func.func private @_QFhost9Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost9Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_10]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_11]](%[[VAL_10]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_12]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  end subroutine
  function return_array()
    use :: some_module
    real :: return_array(n_common)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on a common block variable declared in B.
subroutine host10()
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func.func private @_QFhost10Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  fir.shape %[[SELECT:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]] = hlfir.eval_in_mem shape %[[VAL_8:.*]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:    %[[RES:.*]] = fir.call @_QFhost10Preturn_array() {{.*}}: () -> !fir.array<?xf32>
! CHECK:    fir.save_result %[[RES]] to %[[ARG]](%[[VAL_8]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_10:.*]]:3 = hlfir.associate %[[VAL_9]](%[[VAL_8]]) {{.*}} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
! CHECK:  fir.call @_QPtakes_array(%[[VAL_10]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  end subroutine
  function return_array()
    integer :: n_common
    common /mycom/ n_common
    real :: return_array(n_common)
  end function
end subroutine


! Test call to a function returning an array where the interface is use
! associated from a module.
module define_interface
contains
function foo()
  real :: foo(100)
  foo = 42
end function
end module
! CHECK-LABEL: func.func @_QPtest_call_to_used_interface(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
subroutine test_call_to_used_interface(dummy_proc)
  use define_interface
  procedure(foo) :: dummy_proc
  call takes_array(dummy_proc())
! CHECK:  fir.shape %[[VAL_9:.*]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10:.*]] : (!fir.shape<1>) -> !hlfir.expr<100xf32> {
! CHECK:  ^bb0(%[[ARG:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:    %[[VAL_12:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> !fir.array<100xf32>)
! CHECK:    %[[VAL_13:.*]] = fir.call %[[VAL_12]]() fastmath<contract> : () -> !fir.array<100xf32>
! CHECK:    fir.save_result %[[VAL_13]] to %[[ARG]](%[[VAL_10]]) : !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>, !fir.shape<1>
! CHECK:  }
! CHECK:  %[[VAL_14:.*]]:3 = hlfir.associate %[[VAL_11]](%[[VAL_10]]) {adapt.valuebyref} : (!hlfir.expr<100xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>, i1)
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_14]]#0 : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_15]]) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
end subroutine
