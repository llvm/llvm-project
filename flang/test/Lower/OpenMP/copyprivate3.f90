! Test lowering of COPYPRIVATE with procedure pointers.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHICK-SAME:    %arg0: [[TYPE:!fir.ref<!fir.boxproc<() -> i32>>>]],

!CHECK-LABEL: func.func private @_copy_boxproc_i32_args(
!CHECK-SAME:        %arg0: [[TYPE:!fir.ref<!fir.boxproc<\(\) -> i32>>]],
!CHECK-SAME:        %arg1: [[TYPE]])
!CHECK:         %[[DST:.*]]:2 = hlfir.declare %arg0 {{.*}} : ([[TYPE]]) -> ([[TYPE]], [[TYPE]])
!CHECK:         %[[SRC:.*]]:2 = hlfir.declare %arg1 {{.*}} : ([[TYPE]]) -> ([[TYPE]], [[TYPE]])
!CHECK:         %[[TEMP:.*]] = fir.load %[[SRC]]#0 : [[TYPE]]
!CHECK:         fir.store %[[TEMP]] to %[[DST]]#0 : [[TYPE]]
!CHECK:         return

!CHECK-LABEL: func @_QPtest_proc_ptr
!CHECK:         omp.parallel
!CHECK:           omp.single copyprivate(%{{.*}}#0 -> @_copy_boxproc_i32_args : !fir.ref<!fir.boxproc<() -> i32>>)
subroutine test_proc_ptr()
  interface
     function sub1() bind(c) result(ret)
       use, intrinsic :: iso_c_binding
       integer(c_int) :: ret
     end function sub1
  end interface

  procedure(sub1), pointer, save, bind(c) :: ffunptr
  !$omp threadprivate(ffunptr)

  !$omp parallel
    ffunptr => sub1
    !$omp single
      ffunptr => sub1
    !$omp end single copyprivate(ffunptr)
    if (ffunptr() /= 1) print *, 'err'
  !$omp end parallel
end subroutine

function sub1() bind(c) result(ret)
  use, intrinsic::iso_c_binding
  integer(c_int) :: ret
  ret = 1
end function sub1
