! RUN: %flang_fc1 -Wfatal-errors -pedantic %s 2>&1 | FileCheck %s --check-prefix=CHECK1
! RUN: not %flang_fc1 -pedantic -Werror %s 2>&1 | FileCheck %s --check-prefix=CHECK2
! RUN: not %flang_fc1 -Wfatal-errors -pedantic -Werror %s 2>&1 | FileCheck %s --check-prefix=CHECK3

module m
    contains
    subroutine foo(a)
        real, intent(in), target :: a(:)
    end subroutine
end module

program test
    use m
    real, target :: a(1)
    real :: b(1)
    call foo(a) ! ok
    !CHECK1: fatal-errors-warnings.f90:{{.*}} warning:
    !CHECK2: fatal-errors-warnings.f90:{{.*}} warning:
    !CHECK3: fatal-errors-warnings.f90:{{.*}} warning:
    call foo(b)
    !CHECK1: fatal-errors-warnings.f90:{{.*}} warning:
    !CHECK2: fatal-errors-warnings.f90:{{.*}} warning:
    !CHECK3-NOT: error:
    !CHECK3-NOT: warning:
    call foo((a))
    !CHECK1: fatal-errors-warnings.f90:{{.*}} warning:
    !CHECK2: fatal-errors-warnings.f90:{{.*}} warning:
    call foo(a([1]))
    !! Hard error instead of warning if uncommented.
    !call foo(a(1))
end