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

program main
real a
!CHECK: bar (Subroutine, OmpDeclareTarget): HostAssoc
!CHECK: baz (Function, OmpDeclareTarget): HostAssoc
!$omp declare target(bar)
!$omp declare target(baz)

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
