! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type base
   contains
     procedure, nopass :: tbp
  end type
  type, extends(base), abstract :: child
   contains
     !ERROR: Override of non-DEFERRED 'tbp' must not be DEFERRED
     procedure(tbp), deferred, nopass :: tbp
  end type
 contains
  subroutine tbp
  end
end
