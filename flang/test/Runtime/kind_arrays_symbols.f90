!RUN: %flang -o %t %s

program kind_array_symbols
    use iso_fortran_env, only: integer_kinds, real_kinds, logical_kinds
    implicit none

    integer :: i
    i = 1

    ! accesses via a variable array index cause code-gen
    ! to emit used symbols in the object code
    print *, integer_kinds(i)
    print *, real_kinds(i)
    print *, logical_kinds(i)
end program kind_array_symbols
