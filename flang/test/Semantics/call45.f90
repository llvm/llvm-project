! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
program call45
    integer :: v(100) = [(i, i=1, 100)]
    !ERROR: Actual argument associated with VOLATILE dummy argument 'v=' is not definable [-Wundefinable-asynchronous-or-volatile-actual]
    !BECAUSE: Variable 'v([INTEGER(8)::1_8,2_8,2_8,3_8,3_8,3_8,4_8,4_8,4_8,4_8])' has a vector subscript
    call sub(v([1,2,2,3,3,3,4,4,4,4]))
    !OK: Some compilers don't allow this, but there doesn't seem to be a good reason to disallow it.
    !PORTABILITY: The array section 'v(21_8:30_8:1_8)' may not be associated with dummy argument 'v=' with VOLATILE attribute, unless the dummy is assumed-shape or assumed-rank [-Warray-section-copy-in-copy-out]
    call sub(v(21:30))
    print *, v
contains
    subroutine sub(v)
        integer, volatile :: v(10)
        v = 0
    end subroutine sub
end program call45
