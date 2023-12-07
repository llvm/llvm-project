! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
subroutine s1
  call ext(1, 2)
end

subroutine s2
  !WARNING: Reference to the procedure 'ext' has an implicit interface that is distinct from another reference: distinct numbers of dummy arguments
  call ext(1.)
end

subroutine s3
  interface
    !WARNING: The global subprogram 'ext' is not compatible with its local procedure declaration (incompatible procedure attributes: ImplicitInterface)
    subroutine ext(n)
      integer n
    end
  end interface
  call ext(3)
  !ERROR: Actual argument type 'REAL(4)' is not compatible with dummy argument type 'INTEGER(4)'
  call ext(4.)
end
