! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s

subroutine bar(i, a)
    !$omp declare target
    real :: a
    integer :: i
    a = a - i
end subroutine

function baz(a)
    !$omp declare target
    real, intent(in) :: a
    baz = a
end function baz

subroutine importer()
    external ext_routine
    integer ext_function
    external ext_function
    !$omp declare target(ext_routine)
    !$omp declare target(ext_function)
    !CHECK: ext_function, EXTERNAL (Function, OmpDeclareTarget): ProcEntity {{.*}}
    !CHECK: ext_routine, EXTERNAL (OmpDeclareTarget): ProcEntity
end subroutine importer

program main
real a
external ext_routine
integer ext_function
external ext_function
!CHECK: bar (Subroutine, OmpDeclareTarget): HostAssoc
!CHECK: baz (Function, OmpDeclareTarget): HostAssoc
!CHECK: ext_function, EXTERNAL (Function, OmpDeclareTarget): ProcEntity {{.*}}
!CHECK: ext_routine, EXTERNAL (OmpDeclareTarget): ProcEntity
!$omp declare target(bar)
!$omp declare target(baz)
!$omp declare target(ext_function)
!$omp declare target(ext_routine)

a = baz(a)
call bar(2,a)
call foo(a)
return
end

subroutine foo(a)
real a
integer i
!CHECK: bar (Subroutine, OmpDeclareTarget): HostAssoc
!CHECK: baz (Function, OmpDeclareTarget): HostAssoc
!$omp declare target(bar)
!$omp declare target(baz)
!$omp target
    a = baz(a)
    call bar(i,a)
!$omp end target
return
end
