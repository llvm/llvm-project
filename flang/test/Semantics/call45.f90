! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
program call45
    integer, target :: v(100) = [(i, i=1, 100)]
    integer, pointer :: p(:) => v
    !ERROR: Actual argument associated with VOLATILE dummy argument 'v=' is not definable [-Wundefinable-asynchronous-or-volatile-actual]
    !BECAUSE: Variable 'v([INTEGER(8)::1_8,2_8,2_8,3_8,3_8,3_8,4_8,4_8,4_8,4_8])' has a vector subscript
    call sub(v([1,2,2,3,3,3,4,4,4,4]))
    !PORTABILITY: The array section 'v(21_8:30_8:1_8)' should not be associated with dummy argument 'v=' with VOLATILE attribute, unless the dummy is assumed-shape or assumed-rank [-Wportability]
    call sub(v(21:30))
    !PORTABILITY: The array section 'v(21_8:40_8:2_8)' should not be associated with dummy argument 'v=' with VOLATILE attribute, unless the dummy is assumed-shape or assumed-rank [-Wportability]
    call sub(v(21:40:2))
    call sub2(v(21:40:2))
    call sub4(p)
    print *, v
contains
    subroutine sub(v)
        integer, volatile :: v(10)
        v = 0
    end subroutine sub
    subroutine sub1(v)
        integer, volatile :: v(:)
        v = 0
    end subroutine sub1
    subroutine sub2(v)
        integer :: v(:)
        !TODO: This should either be an portability warning or copy-in-copy-out warning
        call sub(v)
        call sub1(v)
    end subroutine sub2
    subroutine sub3(v)
        integer, pointer :: v(:)
        v = 0
    end subroutine sub3
    subroutine sub4(v)
        integer, pointer :: v(:)
        !TODO: This should either be a portability warning or copy-in-copy-out warning
        call sub(v)
        call sub1(v)
        call sub3(v)
    end subroutine sub4
end program call45
