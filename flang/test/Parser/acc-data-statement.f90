! RUN: not %flang_fc1 -fsyntax-only -fopenacc %s 2>&1 | FileCheck %s
program acc_data_test
    implicit none
    integer :: a(100), b(100), c(100), d(100)
    integer :: i, s ! FIXME: if s is named sum you get semantic errors.

    ! Positive tests

    ! Basic data construct in program body
    !$acc data copy(a, b) create(c)
    a = 1
    b = 2
    c = a + b
    !$acc end data
    print *, "After first data region"

    ! Data construct within IF block
    if (.true.) then
        !$acc data copyout(a)
        a = a + 1
        !$acc end data
        print *, "Inside if block"
    end if

    ! Data construct within DO loop
    do i = 1, 10
        !$acc data present(a)
        a(i) = a(i) * 2
        !$acc end data
        print *, "Loop iteration", i
    end do

    ! Nested data constructs
    !$acc data copyin(a)
    s = 0
    !$acc data copy(s)
    s = s + 1
    !$acc end data
    print *, "After nested data"
    !$acc end data

    ! Negative tests  
    ! Basic data construct in program body
    !$acc data copy(a, b) create(d) bogus()
    !CHECK: acc-data-statement.f90:
    !CHECK-SAME: error: expected end of OpenACC directive
    !CHECK-NEXT: !$acc data copy(a, b) create(d) bogus()
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: !$acc data copy(a, b) create(d) bogus()
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: execution part
    !CHECK-NEXT: !$acc data copy(a, b) create(c)
    !CHECK-NEXT: ^
    a = 1
    b = 2
    d = a + b
!   !$acc end data
    print *, "After first data region"

    ! Data construct within IF block
    if (.true.) then
        !$acc data copyout(a)
        a = a + 1
!       !$acc end data
        print *, "Inside if block"
        !CHECK: acc-data-statement.f90:
        !CHECK-SAME: error: expected OpenACC end block directive
        !CHECK-NEXT: end if
        !CHECK-NEXT: ^ 
        !CHECK-NEXT: in the context: OpenACC construct
        !CHECK-NEXT: !$acc data copyout(a)
        !CHECK-NEXT: ^
        !CHECK-NEXT: in the context: IF construct
        !CHECK-NEXT: if (.true.) then
        !CHECK-NEXT: ^
    end if

    ! Data construct within DO loop
    do i = 1, 10
        !$acc data present(a)
        a(i) = a(i) * 2
!       !$acc end data
        print *, "Loop iteration", i
        !CHECK: acc-data-statement.f90:
        !CHECK-SAME: error: expected OpenACC end block directive
        !CHECK-NEXT: end do
        !CHECK-NEXT: ^ 
        !CHECK-NEXT: in the context: OpenACC construct
        !CHECK-NEXT: !$acc data present(a)
        !CHECK-NEXT: ^
        !CHECK-NEXT: in the context: DO construct
        !CHECK-NEXT: do i = 1, 10
        !CHECK-NEXT: ^
    end do

    ! Nested data constructs
    !$acc data copyin(a)
    s = 0
    !$acc data copy(s)
    s = s + 1
!   !$acc end data
    print *, "After nested data"
    !$acc end data  I forgot to comment this out.
    !CHECK: acc-data-statement.f90:
    !CHECK-SAME: error: expected end of OpenACC directive
    !CHECK-NEXT: !$acc end data  I forgot to comment this out.
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: !$acc data copy(s)
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: !$acc data copyin(a)
    !CHECK-NEXT: ^
    print *, "Program finished"

    !CHECK: acc-data-statement.f90:
    !CHECK-SAME: error: expected OpenACC end block directive
    !CHECK-NEXT: contains
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: !$acc data copyin(a)
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: !$acc data copy(a, b) create(d) bogus()
    !CHECK-NEXT: ^
    !CHECK: acc-data-statement.f90:
    !CHECK-SAME: error: expected OpenACC end block directive
    !CHECK-NEXT: contains
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: OpenACC construct
    !CHECK-NEXT: $acc data copy(a, b) create(d) bogus()
    !CHECK-NEXT: ^
    !CHECK-NEXT: in the context: execution part
    !CHECK-NEXT: !$acc data copy(a, b) create(c)
    !CHECK-NEXT: ^
contains
    subroutine positive_process_array(x)
        integer, intent(inout) :: x(:)
        
        ! Data construct in subroutine
        !$acc data copy(x)
        x = x + 1
        !$acc end data
        print *, "Subroutine finished"
    end subroutine

    function positive_compute_sum(x) result(total)
        integer, intent(in) :: x(:)
        integer :: total
        
        ! Data construct in function
        !$acc data copyin(x) copy(total)
        total = sum(x)
        !$acc end data
        print *, "Function finished"
    end function
    
    subroutine negative_process_array(x)
        integer, intent(inout) :: x(:)
        
        ! Data construct in subroutine
        !$acc data copy(x)
        x = x + 1
!       !$acc end data
        print *, "Subroutine finished"
        !CHECK: acc-data-statement.f90:
        !CHECK-SAME: error: expected OpenACC end block directive
        !CHECK-NEXT: end subroutine
        !CHECK-NEXT: ^
        !CHECK-NEXT: in the context: OpenACC construct
        !CHECK-NEXT: !$acc data copy(x)
        !CHECK-NEXT: ^
        !CHECK-NEXT: in the context: SUBROUTINE subprogram
        !CHECK-NEXT: subroutine negative_process_array(x)
        !CHECK-NEXT: ^
    end subroutine

    function negative_compute_sum(x) result(total)
        integer, intent(in) :: x(:)
        integer :: total
        total = sum(x)
        ! Data construct in function
        !$acc data copyin(x) copy(total)
        total = total + x
!       !$acc end data
        print *, "Function finished"
        !CHECK: acc-data-statement.f90:
        !CHECK-SAME: error: expected OpenACC end block directive
        !CHECK-NEXT: end function
        !CHECK-NEXT: ^ 
        !CHECK-NEXT: in the context: OpenACC construct
        !CHECK-NEXT: !$acc data copyin(x) copy(total)
        !CHECK-NEXT: ^
        !CHECK-NEXT: in the context: execution part
        !CHECK-NEXT: total = sum(x)
        !CHECK-NEXT: ^
    end function
end program acc_data_test