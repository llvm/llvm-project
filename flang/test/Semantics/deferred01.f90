! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Deferred TBPs must be overridden, but when they are private, those
! overrides are required to appear in the same module.  We allow overrides
! elsewhere as an extension.
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
    !WARNING: Override of PRIVATE DEFERRED 'deferredtbp' should appear in its module [-Winaccessible-deferred-override]
    procedure :: deferredTbp => implTbp
  end type
 contains
  subroutine implTbp(x)
    class(ext), intent(in) :: x
  end
end
