! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test for parsing confusion between array indexing and string subscripts

! This is okay: selects the whole substring
subroutine substring_0(c)
  character(:), pointer :: c
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(out:c(:))
  !$omp end task
end

! This is okay: selects from the second character onwards
subroutine substring_1(c)
  character(:), pointer :: c
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(out:c(2:))
  !$omp end task
end

! This is okay: selects the first 2 characters
subroutine substring_2(c)
  character(:), pointer :: c
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !$omp task depend(out:c(:2))
  !$omp end task
end

! Error
subroutine substring_3(c)
  character(:), pointer :: c
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Substrings must be in the form parent-string(lb:ub)
  !$omp task depend(out:c(2))
  !$omp end task
end

! This is okay: interpreted as indexing into the array not as a substring
subroutine substring_3b(c)
  character(:), pointer :: c(:)
  !$omp task depend(out:c(2))
  !$omp end task
end

! This is okay: no indexing or substring at all
subroutine substring_4(c)
  character(:), pointer :: c
  !$omp task depend(out:c)
  !$omp end task
end

! This is not okay: substrings can't have a stride
subroutine substring_5(c)
  character(:), pointer :: c
  !PORTABILITY: The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2.
  !ERROR: Cannot specify a step for a substring
  !$omp task depend(out:c(1:20:5))
  !$omp end task
end

! This is okay: interpreted as indexing the array
subroutine substring_5b(c)
  character(:), pointer :: c(:)
  !$omp task depend(out:c(1:20:5))
  !$omp end task
end
