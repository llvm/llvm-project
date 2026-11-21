! DECLARE SIMD applies to all entries of a subprogram, not just to the main one.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine main_entry(x)
  !$omp declare simd linear(x:1)
  integer :: x, y
  call foo()

entry alt_entry_1(x, y)
  call bar(x, y)
  return

entry alt_entry_2(x)
  call baz(x)
  return
end subroutine

! CHECK-LABEL: func.func @_QPmain_entry(
! CHECK-SAME:  %[[X_ARG:.*]]: !fir.ref<i32>{{.*}})
! CHECK:       %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:       omp.declare_simd linear(%[[X]]
! CHECK:       return

! CHECK-LABEL: func.func @_QPalt_entry_1(
! CHECK-SAME:  %[[X_ARG:.*]]: !fir.ref<i32>{{.*}},{{.*}})
! CHECK:       %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:       omp.declare_simd linear(%[[X]]
! CHECK:       return

! CHECK-LABEL: func.func @_QPalt_entry_2(
! CHECK-SAME:  %[[X_ARG:.*]]: !fir.ref<i32>{{.*}})
! CHECK:       %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:       omp.declare_simd linear(%[[X]]
! CHECK:       return

module mymod
  contains
  subroutine f(x)
    !$omp declare simd linear(x:1)
    integer :: x, y
    call foo()

  entry g(x, y)
    call bar(x, y)
    return
  end subroutine
end module mymod

! CHECK-LABEL: func.func @_QMmymodPf(
! CHECK-SAME:  %[[X_ARG:.*]]: !fir.ref<i32>{{.*}})
! CHECK:       %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:       omp.declare_simd linear(%[[X]]
! CHECK:       return

! CHECK-LABEL: func.func @_QMmymodPg(
! CHECK-SAME:  %[[X_ARG:.*]]: !fir.ref<i32>{{.*}},{{.*}})
! CHECK:       %[[X:.*]]:2 = hlfir.declare %[[X_ARG]]
! CHECK:       omp.declare_simd linear(%[[X]]
! CHECK:       return
