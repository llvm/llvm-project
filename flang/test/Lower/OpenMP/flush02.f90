! This test checks lowering of OpenMP Flush Directive.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module flush02_mod
    type t1
       integer(kind=4) :: x = 4
    end type t1

    type :: t2
       type(t1) :: y = t1(2)
    end type t2


contains

    subroutine sub01(pt)
        class(t1), intent(inout) :: pt
        type(t2)                 :: dt
        integer, allocatable     :: a(:)
        integer, pointer         :: b(:)

        ! CHECK: omp.flush({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
        ! CHECK: omp.flush({{.*}} : !fir.ref<f32>)
        ! CHECK: omp.flush({{.*}} : !fir.ref<!fir.type<_QMflush02_modTt2{y:!fir.type<_QMflush02_modTt1{x:i32}>}>>)
        ! CHECK: omp.flush({{.*}} : !fir.class<!fir.type<_QMflush02_modTt1{x:i32}>>)
        !$omp flush(a)
        !$omp flush(p)
        !$omp flush(dt)
        !$omp flush(pt)
    end subroutine
end module flush02_mod
