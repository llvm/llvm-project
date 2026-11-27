!RUN: %python %S/test_errors.py %s %flang_fc1
interface
  subroutine sub(x)
    real x
    namelist /useless/x ! ok, but don't crash
    !ERROR: 'sub' is not a variable
    namelist /bad/sub
  end
end interface
end
