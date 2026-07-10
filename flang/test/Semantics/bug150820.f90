!RUN: %python %S/test_errors.py %s %flang_fc1
subroutine a
 contains
  subroutine b
   contains
    !ERROR: An internal subprogram may not contain an internal subprogram
    subroutine c
    end
  end
end

program p
 contains
  subroutine b
   contains
    !ERROR: An internal subprogram may not contain an internal subprogram
    subroutine c
    end
  end
end
