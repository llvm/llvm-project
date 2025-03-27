! Tests that locally destroyed values in a `do concurrent` loop are properly
! handled. Locally destroyed values are those values for which the Fortran runtime
! calls `@_FortranADestroy` inside the loops body. If these values are allocated
! outside the loop, and the loop is mapped to OpenMP, then a runtime error would
! occur due to multiple teams trying to access the same allocation.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s --check-prefixes=COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s --check-prefixes=COMMON,DEVICE

module struct_mod
    type test_struct
        integer, allocatable :: x_
    end type

    interface test_struct
        pure module function construct_from_components(x) result(struct)
            implicit none
            integer, intent(in) :: x
            type(test_struct) struct
        end function
    end interface
end module

submodule(struct_mod) struct_sub
    implicit none

contains
    module procedure construct_from_components
        struct%x_ = x
    end procedure
end submodule struct_sub

program main
    use struct_mod, only : test_struct

    implicit none
    type(test_struct), dimension(10) :: a
    integer :: i
    integer :: total

    do concurrent (i=1:10)
        a(i) = test_struct(i)
    end do

    do i=1,10
        total = total + a(i)%x_
    end do

    print *, "total =", total
end program main

! DEVICE: omp.target {{.*}} {
! DEVICE: omp.teams {
! COMMON: omp.parallel {
! COMMON:   %[[LOCAL_TEMP:.*]] = fir.alloca !fir.type<_QMstruct_modTtest_struct{x_:!fir.box<!fir.heap<i32>>}> {bindc_name = ".result"}
! DEVICE:   omp.distribute {
! COMMON:   omp.wsloop {
! COMMON:     omp.loop_nest {{.*}} {
! COMMON:       %[[TEMP_VAL:.*]] = fir.call @_QMstruct_modPconstruct_from_components
! COMMON:       fir.save_result %[[TEMP_VAL]] to %[[LOCAL_TEMP]]
! COMMON:       %[[EMBOXED_LOCAL:.*]] = fir.embox %[[LOCAL_TEMP]]
! COMMON:       %[[CONVERTED_LOCAL:.*]] = fir.convert %[[EMBOXED_LOCAL]]
! COMMON:       fir.call @_FortranADestroy(%[[CONVERTED_LOCAL]])
! COMMON:       omp.yield
! COMMON:     }
! COMMON:   }
! DEVICE:   }
! COMMON:   omp.terminator
! COMMON: }
! DEVICE: omp.terminator
! DEVICE: }
! DEVICE: omp.terminator
! DEVICE: }
