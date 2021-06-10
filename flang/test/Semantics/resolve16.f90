! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
module m
  interface
    subroutine sub0
    end
    !ERROR: A PROCEDURE statement is only allowed in a generic interface block
    procedure :: sub1, sub2
  end interface
contains
  subroutine sub1
  end
  subroutine sub2
  end
end
