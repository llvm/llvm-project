! Test lowering of calls involving assumed shape arrays or arrays with
! VALUE attribute.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_assumed_to_assumed(x)
  interface
    subroutine takes_assumed(x)
      real :: x(:)
    end subroutine
  end interface
  real :: x(:)
  call takes_assumed(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_assumed_to_assumed(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {uniq_name = "_QFtest_assumed_to_assumedEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  fir.call @_QPtakes_assumed(%[[VAL_1]]#0) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()

subroutine test_ptr_to_assumed(p)
  interface
    subroutine takes_assumed(x)
      real :: x(:)
    end subroutine
  end interface
  real, pointer :: p(:)
  call takes_assumed(p)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ptr_to_assumed(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptr_to_assumedEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_assumed(%[[VAL_3]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()

subroutine test_ptr_to_contiguous_assumed(p)
  interface
    subroutine takes_contiguous_assumed(x)
      real, contiguous :: x(:)
    end subroutine
  end interface
  real, pointer :: p(:)
  call takes_contiguous_assumed(p)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ptr_to_contiguous_assumed(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptr_to_contiguous_assumedEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
! CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_contiguous_assumed(%[[VAL_4]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_3]]#0, %[[VAL_3]]#1 to %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()

subroutine test_ptr_to_contiguous_assumed_classstar(p)
  interface
    subroutine takes_contiguous_assumed_classstar(x)
      class(*), contiguous :: x(:)
    end subroutine
  end interface
  real, pointer :: p(:)
  call takes_contiguous_assumed_classstar(p)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ptr_to_contiguous_assumed_classstar(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptr_to_contiguous_assumed_classstarEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.copy_in %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1)
! CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]]#0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:  fir.call @_QPtakes_contiguous_assumed_classstar(%[[VAL_4]]) {{.*}} : (!fir.class<!fir.array<?xnone>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_3]]#0, %[[VAL_3]]#1 to %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, i1, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> ()

subroutine test_ptr_to_assumed_typestar(p)
  interface
    subroutine takes_assumed_typestar(x)
      type(*) :: x(:)
    end subroutine
  end interface
  real, pointer :: p(:)
  call takes_assumed_typestar(p)
end subroutine
! CHECK-LABEL: func.func @_QPtest_ptr_to_assumed_typestar(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptr_to_assumed_typestarEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<!fir.array<?xnone>>
! CHECK:  fir.call @_QPtakes_assumed_typestar(%[[VAL_3]]) {{.*}} : (!fir.box<!fir.array<?xnone>>) -> ()

subroutine test_explicit_char_to_box(e)
  interface
    subroutine takes_assumed_character(x)
      character(*) :: x(:)
    end subroutine
  end interface
  character(10) :: e(20)
  call takes_assumed_character(e)
end subroutine
! CHECK-LABEL: func.func @_QPtest_explicit_char_to_box(
! CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,10>>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_5:[a-z0-9]*]]) typeparams %[[VAL_2:[a-z0-9]*]] {uniq_name = "_QFtest_explicit_char_to_boxEe"} : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.ref<!fir.array<20x!fir.char<1,10>>>)
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[VAL_5]]) : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.char<1,10>>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<20x!fir.char<1,10>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:  fir.call @_QPtakes_assumed_character(%[[VAL_8]]) {{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()

subroutine test_explicit_by_val(x)
  interface
    subroutine takes_explicit_by_value(x)
      real, value :: x(10)
    end subroutine
  end interface
  real :: x(10)
  call takes_explicit_by_value(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_explicit_by_val(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]]) {uniq_name = "_QFtest_explicit_by_valEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:  %[[VAL_4:.*]] = hlfir.as_expr %[[VAL_3]]#0 : (!fir.ref<!fir.array<10xf32>>) -> !hlfir.expr<10xf32>
! CHECK:  %[[VAL_5:.*]]:3 = hlfir.associate %[[VAL_4]](%[[VAL_2]]) {adapt.valuebyref} : (!hlfir.expr<10xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>, i1)
! CHECK:  fir.call @_QPtakes_explicit_by_value(%[[VAL_5]]#1) {{.*}} : (!fir.ref<!fir.array<10xf32>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.array<10xf32>>, i1
