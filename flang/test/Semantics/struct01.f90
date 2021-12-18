! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for new semantic errors from misuse of the DEC STRUCTURE extension
program main
  !ERROR: Derived type '/undeclared/' not found
  record /undeclared/ var
  structure /s/
    !ERROR: /s/ is not a known STRUCTURE
    record /s/ attemptToRecurse
    !ERROR: UNION is not yet supported
    union
      map
        integer j
      end map
      map
        real x
      end map
    end union
  end structure
end
