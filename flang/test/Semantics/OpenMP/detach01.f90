! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52

! OpenMP Version 5.2: 12.5.2
! Various checks for DETACH Clause

program detach01
    use omp_lib, only: omp_event_handle_kind
    implicit none
    real                                    :: e, x
    integer(omp_event_handle_kind)          :: event_01, event_02(2)
    integer(omp_event_handle_kind), pointer :: event_03

    type :: t
        integer(omp_event_handle_kind) :: event
    end type
    type(t) :: t_01

    !ERROR: The event-handle: `e` must be of type integer(kind=omp_event_handle_kind)
    !$omp task detach(e)
        x = x + 1
    !$omp end task

    !ERROR: If a DETACH clause appears on a directive, then the encountering task must not be a FINAL task
    !$omp task detach(event_01) final(.false.)
        x = x + 1
    !$omp end task

    !ERROR: A variable: `event_01` that appears in a DETACH clause cannot appear on PRIVATE clause on the same construct
    !$omp task detach(event_01) private(event_01)
        x = x + 1
    !$omp end task

    !ERROR: A variable: `event_01` that appears in a DETACH clause cannot appear on FIRSTPRIVATE clause on the same construct
    !$omp task detach(event_01) firstprivate(event_01)
        x = x + 1
    !$omp end task

    !ERROR: A variable: `event_01` that appears in a DETACH clause cannot appear on SHARED clause on the same construct
    !$omp task detach(event_01) shared(event_01)
        x = x + 1
    !$omp end task

    !ERROR: A variable: `event_01` that appears in a DETACH clause cannot appear on IN_REDUCTION clause on the same construct
    !$omp task detach(event_01) in_reduction(+:event_01)
        x = x + 1
    !$omp end task

    !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a DETACH clause
    !$omp task detach(event_02(1))
        x = x + 1
    !$omp end task

    !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a DETACH clause
    !$omp task detach(t_01%event)
        x = x + 1
    !$omp end task

    !ERROR: The event-handle: `event_03` must not have the POINTER attribute
    !$omp task detach(event_03)
        x = x + 1
    !$omp end task
end program
