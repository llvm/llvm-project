! RUN: %python %S/test_errors.py %s %flang_fc1
! Deferred TBPs must be overridden, but when they are private, those
! overrides must appear in the same module.
module m1
  type, abstract :: absBase
   contains
    procedure(deferredInterface), deferred, private :: deferredTbp
  end type
  abstract interface
    subroutine deferredInterface(x)
      import absBase
      class(absBase), intent(in) :: x
    end
  end interface
end

module m2
  use m1
  type, extends(absBase) :: ext
   contains
    !ERROR: Override of PRIVATE DEFERRED 'deferredtbp' must appear in its module
    procedure :: deferredTbp => implTbp
  end type
 contains
  subroutine implTbp(x)
    class(ext), intent(in) :: x
  end
end
